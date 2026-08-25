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
