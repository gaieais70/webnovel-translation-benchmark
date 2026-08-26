"""OpenAI-compatible HTTP runner. Used for OpenRouter and direct providers."""

from __future__ import annotations

import time

import httpx

from ati.runners.base import RunResult, looks_like_refusal

# A full web-novel chapter is several thousand source characters, and a
# reasoning model may think for minutes before emitting any of the
# translation. The old 300s default cut legitimate work off mid-generation
# and recorded it as a failure.
TIMEOUT_S = 1200.0


class ApiRunner:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        temperature: float,
        client: httpx.Client | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout_s)

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
        try:
            response = self._client.post(
                self._endpoint, json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            return RunResult(
                text="", latency_s=time.monotonic() - started, error=str(exc)
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
