import json
import httpx
import pytest

from ati.runners.api import ApiRunner
from ati.runners.base import RunResult, looks_like_refusal
from ati.runners.fake import FakeRunner


def test_fake_runner_returns_queued_responses():
    r = FakeRunner(responses=["first", "second"])
    assert r.run("sys", "src").text == "first"
    assert r.run("sys", "src").text == "second"


def test_fake_runner_records_calls():
    r = FakeRunner(responses=["x"])
    r.run("SYSTEM", "SOURCE")
    assert r.calls == [("SYSTEM", "SOURCE")]


def test_fake_runner_can_simulate_error():
    r = FakeRunner(error="boom")
    result = r.run("sys", "src")
    assert result.error == "boom"
    assert result.text == ""


def test_looks_like_refusal_detects_common_phrasings():
    assert looks_like_refusal("I can't help with that request.")
    assert looks_like_refusal("I'm unable to translate this content.")
    assert not looks_like_refusal("He drew his sword and cut the man down.")


def test_api_runner_parses_a_successful_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "translated text"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.0012,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner(
        endpoint="https://example.invalid/v1/chat/completions",
        model="vendor/model",
        api_key="k",
        temperature=0.3,
        client=client,
    )
    result = runner.run("sys", "src")
    assert isinstance(result, RunResult)
    assert result.text == "translated text"
    assert result.tokens_in == 100
    assert result.tokens_out == 50
    assert result.cost_usd == 0.0012
    assert result.error is None
    assert result.latency_s >= 0


def test_api_runner_reports_cost_none_when_usage_omits_it():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("sys", "src")
    assert result.cost_usd is None


def test_api_runner_records_http_error_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("sys", "src")
    assert result.error is not None
    assert "429" in result.error
    assert result.text == ""


def test_api_runner_flags_refusals():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I can't help with that."}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("sys", "src").refused is True


def test_api_runner_reads_top_level_cost():
    """opencode-go returns cost outside usage, unlike OpenRouter."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "cost": 0.0031,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").cost_usd == 0.0031


def test_api_runner_prefers_usage_cost_over_top_level():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "cost": 9.99,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").cost_usd == 0.5


def test_api_runner_keeps_zero_cost_distinct_from_unknown():
    """Subscription access bills zero. That is not the same as unknown."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "t"}}], "cost": 0},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").cost_usd == 0.0


def test_api_runner_captures_reasoning_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "usage": {
                    "prompt_tokens": 43,
                    "completion_tokens": 178,
                    "completion_tokens_details": {"reasoning_tokens": 153},
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.reasoning_tokens == 153
    assert result.tokens_out == 178


def test_api_runner_reasoning_tokens_none_when_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "t"}}], "usage": {}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").reasoning_tokens is None


def test_api_runner_surfaces_a_read_timeout_without_retrying():
    """Superseded an earlier test that retried read timeouts.

    Retrying one meant paying for the same runaway generation up to three
    times, which is how twelve 65,536-token calls were billed.
    """
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ReadTimeout("still generating", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    result = runner.run("s", "u")
    assert attempts["n"] == 1
    assert result.error is not None


def test_api_runner_gives_up_after_the_attempt_limit():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner(
        "http://x", "m", "k", 0.3, client=client, attempts=3, retry_backoff_s=0
    )
    result = runner.run("s", "u")
    assert attempts["n"] == 3
    assert "failed after 3 attempts" in result.error
    assert result.text == ""


def test_api_runner_does_not_retry_a_successful_call():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    runner.run("s", "u")
    assert attempts["n"] == 1


def test_output_budget_scales_with_source_and_is_clamped():
    from ati.runners.api import (
        MAX_OUTPUT_TOKENS,
        REASONING_HEADROOM_TOKENS,
        output_token_budget,
    )

    # the reasoning allowance is the floor; it does not scale with the source
    assert output_token_budget("") == REASONING_HEADROOM_TOKENS
    assert output_token_budget("x" * 100) > REASONING_HEADROOM_TOKENS
    # translation estimate (11,100) plus the fixed reasoning allowance
    assert output_token_budget("x" * 7400) == 24576
    assert output_token_budget("x" * 10_000_000) == MAX_OUTPUT_TOKENS


def test_api_runner_sends_a_max_tokens_ceiling():
    """Without this a model can generate to the provider cap: 65,536 tokens
    at $0.29 a call, against $0.02 for the same work bounded."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    runner.run("s", "x" * 7400)
    assert seen["max_tokens"] == 24576


def test_api_runner_max_tokens_can_be_overridden():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, max_output_tokens=500)
    runner.run("s", "x" * 7400)
    assert seen["max_tokens"] == 500


def test_api_runner_records_truncation_as_a_failure():
    """A translation cut off at the ceiling is incomplete, not a bad answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "half a trans"}, "finish_reason": "length"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 65536},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.error is not None
    assert "truncated" in result.error
    assert result.text == ""
    assert result.tokens_out == 65536


def test_api_runner_keeps_a_normally_finished_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "complete"}, "finish_reason": "stop"}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.text == "complete"
    assert result.error is None


def test_api_runner_does_not_retry_a_read_timeout():
    """The server may still be generating; a retry pays for it again."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ReadTimeout("still generating", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    result = runner.run("s", "u")
    assert attempts["n"] == 1
    assert "ReadTimeout" in result.error


def test_api_runner_still_retries_connection_failures():
    """These never reached a model, so repeating them costs nothing."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    result = runner.run("s", "u")
    assert attempts["n"] == 3
    assert result.text == "ok"


def test_budget_covers_observed_reasoning_on_a_short_source():
    """An 800-char passage was seen using 15,312 completion tokens, nearly all
    hidden reasoning. Budgeting from source length alone would truncate it."""
    from ati.runners.api import output_token_budget

    assert output_token_budget("x" * 800) > 15312


def test_budget_still_blocks_the_runaway_ceiling():
    from ati.runners.api import MAX_OUTPUT_TOKENS, output_token_budget

    assert output_token_budget("x" * 100_000) == MAX_OUTPUT_TOKENS
    assert MAX_OUTPUT_TOKENS < 65536


def test_truncated_result_keeps_the_refusal_flag():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "I can't help with that request"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.error is not None
    assert result.refused is True


def test_non_dict_choice_is_recorded_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": ["not a dict"]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.error is not None
    assert "malformed response" in result.error
