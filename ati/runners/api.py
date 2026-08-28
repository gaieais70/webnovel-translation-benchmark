"""OpenAI-compatible HTTP runner. Used for OpenRouter and direct providers."""

from __future__ import annotations

import time

import httpx

from ati.runners.base import RunResult, looks_like_refusal

# Measured across models on real chapters: 190-240s is normal, and the fastest
# observed was 60s. Because max_tokens now bounds the response, the worst case
# is bounded too - roughly MAX_OUTPUT_TOKENS at observed throughput - so a
# generous timeout no longer risks waiting out a runaway. That bound is what
# sets this number: glm-5.3-flash sustained 60-72 tokens/sec and spent 573s on
# a single chapter, so a call reaching the clamp needs ~820s and 600 would cut
# it off mid-generation.
TIMEOUT_S = 900.0
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

# Without an explicit ceiling a model can generate until it reaches the
# provider's own cap. Observed: twelve calls that each ran to exactly 65,536
# output tokens and cost $0.29, against $0.02 for the same work done
# correctly. A translation cannot legitimately be much longer than its source,
# so the budget is derived from the input and clamped.
# Measured, not guessed. Across the smoke run hy3 spent 3.05 output tokens per
# source character on a 7,308-character chapter and 3.58 on a 3,625-character
# one; glm-5.3-flash exceeded 3.74 and was cut off. At 1.5 the budget only
# cleared those runs because the flat reasoning headroom below absorbed the
# difference, and that margin vanishes on the longer chapters: chapter 3 is
# 8,705 characters and left hy3 at ~90% of budget. 2.5 keeps the ceiling a real
# bound on runaway generation while no longer scoring a verbose-but-honest
# model as a failure.
TOKENS_PER_SOURCE_CHAR = 2.5
# glm-5.3-flash truncated at exactly 27,346 tokens on chapter 1 and was dropped
# from the ranking for it - a budget artifact reported as a model failure. The
# clamp is raised so the longest chapter in the corpus (8,705 characters, a
# 38,146-token budget) fits without it binding, and still sits a third below
# the 65,536 provider limit that produced the runaway bills.
MAX_OUTPUT_TOKENS = 49152
# Hidden reasoning counts toward completion_tokens, and therefore toward
# max_tokens, but it does not scale with the source: an 800-character passage
# was observed spending 15,312 completion tokens with only ~450 of visible
# translation. Budgeting from source length alone would truncate every short
# task on a reasoning model. This allowance is added on top, and the clamp
# still sits far below the provider ceiling of 65,536 that produced the
# runaway bills.
# Raised from 16,384 after glm-5.3-flash truncated on chapter 2 - the
# second-shortest passage in the corpus - having spent 25,446 completion
# tokens where the same model finished the same chapter in 15,977 during the
# smoke run. Hidden reasoning varies by that much call to call and does not
# shrink with the source, so a flat allowance sized for the average starves
# the short passages first: chapter 2 draws only 9,062 tokens from its own
# length and everything else from here.
REASONING_HEADROOM_TOKENS = 24576


def output_token_budget(source_text: str) -> int:
    """A ceiling covering the translation plus a fixed reasoning allowance.

    Generous by design - the point is to stop runaway generation, not to
    truncate honest work. A 7,400-character chapter needed ~3,600 visible
    tokens but 22,323 completion tokens once hidden reasoning is counted; a
    short passage on the same model needed ~15,000 mostly-hidden ones. Both
    fit under this ceiling; 65,536 does not.
    """
    estimate = int(len(source_text) * TOKENS_PER_SOURCE_CHAR)
    return min(MAX_OUTPUT_TOKENS, estimate + REASONING_HEADROOM_TOKENS)


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
        max_output_tokens: int | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout_s)
        self._attempts = max(1, attempts)
        self._retry_backoff_s = retry_backoff_s
        self._max_output_tokens = max_output_tokens

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
                response = self._client.post(
                    self._endpoint, json=payload, headers=headers
                )
                if response.status_code in RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt + 1 < self._attempts:
                        time.sleep(self._retry_backoff_s * (attempt + 1))
                        continue
                    # Out of attempts: hand back the response so the caller
                    # reports the real status rather than a retry summary.
                return response, None
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self._attempts:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
            except httpx.HTTPError as exc:
                return None, f"{type(exc).__name__}: {exc}"
        return None, f"failed after {self._attempts} attempts: {last_error}"

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens
            or output_token_budget(source_text),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
        }
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
            body = response.json()
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

        result = RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            refused=looks_like_refusal(text),
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
                error=(
                    f"truncated at the output limit "
                    f"({payload['max_tokens']} tokens); translation incomplete"
                ),
            )
        return result
