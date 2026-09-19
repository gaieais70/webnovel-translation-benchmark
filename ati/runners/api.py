"""OpenAI-compatible HTTP runner. Used for OpenRouter and direct providers."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import httpx

from ati.runners.base import RunResult, looks_like_refusal

# Measured across models on real chapters: 190-240s is normal at default
# effort, and the fastest observed was 60s. This is set for the worst case
# instead, because the two failure modes it sits between are asymmetric.
#
# Too low loses real work and is indistinguishable from the model failing:
# glm-5.3-flash lost ch03 and ch08 to a 900s read timeout, and a read timeout
# is deliberately not retried (see `_post`), so the chapter is simply gone. Too
# high costs only wall clock on a call that was never going to return.
#
# With no output ceiling and every model at maximum reasoning effort there is
# no token bound to derive a timeout from, so it is derived from observed
# throughput and the longest run seen: glm-5.3-flash sustains 60-72 tokens/sec
# and spent 1,122s on a single chapter at high effort. 3600 leaves room for a
# max-effort call on the longest passage without cutting it off mid-generation.
TIMEOUT_S = 3600.0
ATTEMPTS = 3
RETRY_BACKOFF_S = 5.0
# A 5xx is the server saying it failed before producing anything, so redoing
# the call costs nothing - the same reasoning that makes connection errors
# safe to repeat. Without this a single blip marks a chapter failed for the
# whole run, and a model that trips two of them across eleven chapters lands
# in the Incomplete table as though the model were at fault. 429 is
# deliberately excluded: that is the usage limit, which the run stage detects
# and stops on cleanly, and hammering it would only burn the window faster.
# 524 is Cloudflare's origin-timeout code, which Command Code fronts its API
# with. It answers `{"type": "server_error", "message": "Upstream model
# provider is temporarily unavailable. Please try again in a moment."}` - the
# gateway saying the provider never produced anything, so a retry is safe and
# is what the response explicitly asks for. Six consecutive anchor calls were
# lost to it before it was retried.
RETRY_STATUS = (500, 502, 503, 504, 524)

# Not every 429 is a usage limit. Command Code answers 429 with "Upstream model
# provider is temporarily unavailable. Please try again in a moment." - the same
# body it sends with 524, and the same meaning: the gateway never reached a
# model. tencent/hy4-preview returned exactly this on both probe calls.
#
# The distinction matters because the run stage stops the WHOLE run on a usage
# limit, which is correct for a real quota but throws away every model still
# queued when it is only a blip. A 429 carrying one of these markers is treated
# as transient - retried here, and reported so `is_usage_limit` does not match.
TRANSIENT_429_MARKERS = (
    "temporarily unavailable",
    "try again",
    "overloaded",
    "no instances available",
    "upstream",
)


def is_transient_429(body: str) -> bool:
    """Whether a 429 body says 'busy right now' rather than 'out of quota'."""
    lowered = (body or "").lower()
    return any(marker in lowered for marker in TRANSIENT_429_MARKERS)

# No output ceiling is sent by default. An earlier version derived one from
# source length (2.5 tokens per character plus a flat reasoning allowance,
# clamped at 49,152) to stop runaway generation after twelve calls each ran to
# 65,536 tokens and cost $0.29 against $0.02 for the same work done correctly.
#
# That ceiling was retired because it destroyed more than it protected. Cost is
# no longer the constraint - a full eleven-model run spent $0.75, all of it on
# the one OpenRouter route, because every other provider here is subscription -
# while the ceiling cost four chapters outright. Reasoning is emitted before
# visible content, so a budget that binds does not shorten a translation, it
# removes it: every truncated call in the 2026-08 run recorded a full token
# spend and zero characters of English.
#
# Two guards cover what the ceiling was actually for, and both act on the
# output text rather than a token proxy: `ran_away` and `came_up_short` in
# ati/runstage.py measure the expansion ratio against the source. Omitting
# max_tokens hands the limit to the provider's own default, which on these
# routes is the model ceiling - that is what the original 65,536-token runaways
# were hitting with no ceiling sent. `finish_reason` is still checked below, so
# if a provider default ever does bind it is recorded rather than inferred.
#
# A per-model ceiling remains available through `params` in config/models.yaml,
# which is how a metered route can still be capped.


def assemble_stream(raw: str) -> dict:
    """Rebuild a chat-completion body from a server-sent-event stream.

    Returns the same shape a non-streaming call does, so everything downstream
    - the finish_reason check, usage and cost, the refusal heuristic - reads a
    streamed answer exactly as it reads a buffered one. A stream that carries an
    error, or that stops without a finish reason or a [DONE] marker, comes back
    as an error body: a translation cut off mid-stream is not a translation.
    """
    content: list[str] = []
    finish = None
    usage = None
    cost = None
    error = None
    native = None
    done = False
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue  # blank separators and ": keep-alive" comments
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("error"):
            error = chunk["error"]
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("cost") is not None:
            cost = chunk["cost"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if choice.get("native_finish_reason"):
                native = choice["native_finish_reason"]
    if error is not None:
        return {"error": error}
    if finish is None and not done:
        return {"error": "stream ended before the model finished"}
    choice_out = {"message": {"content": "".join(content)}, "finish_reason": finish}
    if native is not None:
        choice_out["native_finish_reason"] = native
    body: dict = {"choices": [choice_out]}
    if usage is not None:
        body["usage"] = usage
    if cost is not None:
        body["cost"] = cost
    return body


class ApiRunner:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        temperature: float,
        client: httpx.Client | None = None,
        timeout_s: float = TIMEOUT_S,
        attempts: int = ATTEMPTS,
        retry_backoff_s: float = RETRY_BACKOFF_S,
        extra_params: dict | None = None,
        fallback: "ApiRunner | None" = None,
        stream: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout_s)
        self._attempts = max(1, attempts)
        self._retry_backoff_s = retry_backoff_s
        self._extra_params = dict(extra_params or {})
        self._fallback = fallback
        # Command Code's gateway abandons a buffered request that has not
        # completed within about two minutes, answering 524 "Upstream model
        # provider is temporarily unavailable". Measured on run 2026-09: every
        # 524 arrived ~125s into its attempt, and every call that succeeded
        # there finished inside that window, so a model that reasons longer at
        # maximum effort could never succeed however often it was retried.
        # Streaming keeps bytes moving for the whole generation.
        self._stream = stream

    def _post(self, payload: dict, headers: dict):
        """Send the request, retrying only failures that cost nothing to redo.

        A read timeout is NOT retried. The server may still be generating, and
        a retry then pays for the same runaway output a second and third time.
        Connection-level failures never reached a model, so they are safe to
        repeat, and so is a 5xx: the server answered, and its answer was that
        it produced nothing.
        """
        last_error: str | None = None
        for attempt in range(self._attempts):
            try:
                if payload.get("stream"):
                    # Read to the end inside the stream so the connection stays
                    # active for the whole generation; afterwards the response
                    # behaves like a buffered one for the checks below.
                    with self._client.stream(
                        "POST", self._endpoint, json=payload, headers=headers
                    ) as response:
                        response.read()
                else:
                    response = self._client.post(
                        self._endpoint, json=payload, headers=headers
                    )
                transient = response.status_code == 429 and is_transient_429(
                    response.text
                )
                # A stream the gateway closed early arrives as a clean 200 with
                # no finish reason and no [DONE]. That is a dropped connection
                # in all but name - glm-5.3-flash lost one 78 seconds in - so
                # it is retried like one.
                cut_short = False
                if payload.get("stream") and response.status_code == 200:
                    streamed = assemble_stream(response.text)
                    cut_short = "stream ended" in str(streamed.get("error", ""))
                if response.status_code in RETRY_STATUS or transient or cut_short:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt + 1 < self._attempts:
                        time.sleep(self._retry_backoff_s * (attempt + 1))
                        continue
                    # Out of attempts: hand back the response so the caller
                    # reports the real status rather than a retry summary.
                return response, None
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                # The server hung up - before answering, or partway through a
                # stream. Unlike a read timeout, nothing is still generating
                # on the far side, so another attempt does not pay twice.
                httpx.RemoteProtocolError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self._attempts:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
            except httpx.HTTPError as exc:
                return None, f"{type(exc).__name__}: {exc}"
        return None, f"failed after {self._attempts} attempts: {last_error}"

    def _run_once(self, system_prompt: str, source_text: str) -> RunResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
        }
        # No max_tokens unless the roster row sets one in `params`; absent, the
        # provider's own default applies. Merged last so a roster entry can set
        # a ceiling for one model without a code change. `model` and
        # `messages` are rejected at config load, so nothing here can redirect
        # the request off its own passage.
        payload.update(self._extra_params)
        if self._stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        headers = {"Authorization": f"Bearer {self._api_key}"}
        started = time.monotonic()

        response, error = self._post(payload, headers)
        latency = time.monotonic() - started
        if response is None:
            return RunResult(text="", latency_s=latency, error=error)

        if response.status_code != 200:
            return RunResult(
                text="",
                latency_s=latency,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        try:
            body = assemble_stream(response.text) if self._stream else response.json()
        except ValueError as exc:
            return RunResult(
                text="", latency_s=latency, error=f"malformed response: {exc}"
            )
        # OpenRouter's free routes sometimes answer 200 with an error object
        # in place of choices - the upstream failed after the gateway had
        # already committed to a status. That is the provider failing, not the
        # model answering badly, so it is reported with the provider's own
        # message and treated as a route failure below.
        if isinstance(body, dict) and "choices" not in body and body.get("error"):
            return RunResult(
                text="",
                latency_s=latency,
                error=f"provider error in a 200 response: {str(body['error'])[:200]}",
            )
        try:
            choice = body["choices"][0]
            # content is null, not absent, when a model spends its whole
            # budget on hidden reasoning and emits nothing visible.
            # glm-5.3-flash does this through Command Code. Coerce here: an
            # empty translation is a result the guards below can classify,
            # while None reaches looks_like_refusal and raises TypeError out
            # of run_model, killing every model still queued behind it.
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
            return RunResult(
                text="", latency_s=latency, error=f"malformed response: {exc}"
            )

        usage = body.get("usage") or {}
        # OpenRouter reports cost inside usage; opencode-go reports it at the
        # top level. Zero is a real value on subscription access and must not
        # collapse into None, which means "unknown".
        cost = usage.get("cost")
        if cost is None:
            cost = body.get("cost")
        details = usage.get("completion_tokens_details") or {}

        finish = choice.get("finish_reason")
        result = RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            refused=looks_like_refusal(text),
            finish_reason=finish,
        )

        # The provider's own content filter stopped the answer. That is a
        # refusal as far as a translator is concerned - the route will not
        # translate this - and what was produced before the filter tripped is
        # a fragment, not a translation, so none of it is scored.
        if finish == "content_filter":
            return replace(result, text="", refused=True)

        # The provider ended the generation on its side - a crashed upstream,
        # a wall-clock limit. What arrived is a fragment, and the fault is the
        # route's, so it counts as a route failure and may use the fallback.
        if finish == "error":
            return replace(
                result,
                text="",
                error="provider ended the generation (finish_reason=error); "
                "translation incomplete",
            )

        # A response cut off at the token ceiling is an incomplete translation.
        # Scoring it would penalize the model on fidelity for text it was never
        # allowed to finish, so record it as a failure instead.
        if choice.get("finish_reason") == "length":
            return RunResult(
                text="",
                latency_s=latency,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                reasoning_tokens=result.reasoning_tokens,
                cost_usd=result.cost_usd,
                refused=result.refused,
                finish_reason="length",
                error=(
                    f"truncated at the output limit "
                    f"({payload.get('max_tokens') or 'provider default'} "
                    f"tokens); translation incomplete"
                ),
            )
        return result

    # A route failure is the provider refusing to serve, not a model producing
    # a poor translation. Only the first kind is worth a second route: a
    # refusal, a short answer or a runaway is the model's own output and would
    # be identical on the paid tier, so retrying it there would buy nothing and
    # spend money to do it.
    #
    # A read timeout is included even though `_post` will not retry it on the
    # same route. The reason not to retry is that the server may still be
    # generating and a repeat pays twice for the same runaway - true for one
    # route, but the fallback is a different provider, and on a free route
    # there was nothing to pay in the first place.
    def _is_route_failure(self, result: RunResult) -> bool:
        if not result.error:
            return False
        lowered = result.error.lower()
        if "timeout" in lowered or "timed out" in lowered:
            return True
        if "provider error in a 200 response" in lowered:
            return True
        if "finish_reason=error" in lowered:
            return True
        if "http " not in lowered:
            return "failed after" in lowered  # connection errors, out of tries
        return any(
            f"http {code}" in lowered for code in (*RETRY_STATUS, 429, 402, 403, 404)
        )

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        """Try this route, then the fallback if the route itself failed.

        `route` on the returned result names the slug that actually answered,
        so a model served partly by each is visible in the run record rather
        than silently averaged across two providers.
        """
        result = self._run_once(system_prompt, source_text)
        if result.route is None:
            result = replace(result, route=self._model)
        if self._fallback is None or not self._is_route_failure(result):
            return result

        fallback_result = self._fallback.run(system_prompt, source_text)
        if fallback_result.error:
            # Both routes failed. Report both, so the record does not blame the
            # fallback for a failure that started upstream of it.
            return replace(
                fallback_result,
                error=(
                    f"{self._model}: {result.error} | "
                    f"fallback {self._fallback._model}: {fallback_result.error}"
                ),
            )
        return fallback_result
