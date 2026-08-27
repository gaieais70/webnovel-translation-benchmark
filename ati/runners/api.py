"""OpenAI-compatible HTTP runner. Used for OpenRouter and direct providers."""

from __future__ import annotations

import time

import httpx

from ati.runners.base import RunResult, looks_like_refusal

# Measured on real chapters: ~6,000 input tokens produce ~3,600 output tokens
# in about 60 seconds. 240s therefore leaves generous headroom for a bounded
# call.
TIMEOUT_S = 240.0
ATTEMPTS = 3
RETRY_BACKOFF_S = 5.0

# Without an explicit ceiling a model can generate until it reaches the
# provider's own cap. Observed: twelve calls that each ran to exactly 65,536
# output tokens and cost $0.29, against $0.02 for the same work done
# correctly. A translation cannot legitimately be much longer than its source,
# so the budget is derived from the input and clamped.
TOKENS_PER_SOURCE_CHAR = 1.5
MAX_OUTPUT_TOKENS = 24576
# Hidden reasoning counts toward completion_tokens, and therefore toward
# max_tokens, but it does not scale with the source: an 800-character passage
# was observed spending 15,312 completion tokens with only ~450 of visible
# translation. Budgeting from source length alone would truncate every short
# task on a reasoning model. This allowance is added on top, and the clamp
# still sits far below the provider ceiling of 65,536 that produced the
# runaway bills.
REASONING_HEADROOM_TOKENS = 16384


def output_token_budget(source_text: str) -> int:
    """A ceiling covering the translation plus a fixed reasoning allowance.

    Generous by design - the point is to stop runaway generation, not to
    truncate honest work. A 7,400-character chapter needed ~3,600 visible
    tokens; a short passage on the same model needed ~15,000 mostly-hidden
    ones. Both fit under this ceiling; 65,536 does not.
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
        repeat.
        """
        last_error: str | None = None
        for attempt in range(self._attempts):
            try:
                return self._client.post(self._endpoint, json=payload, headers=headers), None
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
            text = choice["message"]["content"]
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
