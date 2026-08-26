"""OpenAI-compatible HTTP runner. Used for OpenRouter and direct providers."""

from __future__ import annotations

import time

import httpx

from ati.runners.base import RunResult, looks_like_refusal

# Measured: a full 7.4k-character chapter returns in 60-90s. The timeout is
# set well above that but not enormous, because the real failure mode is a
# hung connection rather than a slow one, and hangs are recovered by retrying,
# not by waiting.
TIMEOUT_S = 240.0
ATTEMPTS = 3
RETRY_BACKOFF_S = 5.0


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
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout_s)
        self._attempts = max(1, attempts)
        self._retry_backoff_s = retry_backoff_s

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        started = time.monotonic()
        # The endpoint intermittently accepts a request and never answers.
        # Measured successful calls return in 60-90s, so a connection still
        # silent at the timeout is hung, not slow — waiting longer never
        # rescues it. Retry instead, and give up after a few attempts so one
        # bad model cannot stall the whole run.
        last_error: str | None = None
        for attempt in range(self._attempts):
            try:
                response = self._client.post(
                    self._endpoint, json=payload, headers=headers
                )
                break
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self._attempts:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
        else:
            return RunResult(
                text="",
                latency_s=time.monotonic() - started,
                error=f"failed after {self._attempts} attempts: {last_error}",
            )
        latency = time.monotonic() - started

        if response.status_code != 200:
            return RunResult(
                text="",
                latency_s=latency,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
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
        return RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            refused=looks_like_refusal(text),
        )
