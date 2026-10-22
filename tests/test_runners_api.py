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


def test_api_runner_retries_a_server_error():
    """A 5xx means the server produced nothing, so redoing it costs nothing.

    Previously any 5xx was permanent: one blip marked that chapter failed for
    the whole run, and enough of them put a working model in the Incomplete
    table as though the model were at fault.
    """
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="upstream unavailable")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    result = runner.run("s", "u")
    assert attempts["n"] == 3
    assert result.error is None
    assert result.text == "ok"


def test_api_runner_reports_the_real_status_when_a_server_error_persists():
    """Exhausting the retries must still name the status the server returned.

    muse-spark-1.2-contributor answered 500 to every call; a run log saying
    only "failed after 3 attempts" would not distinguish that from the model
    being unreachable.
    """
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500, text='{"error":"Internal server error"}')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner(
        "http://x", "m", "k", 0.3, client=client, attempts=3, retry_backoff_s=0
    )
    result = runner.run("s", "u")
    assert attempts["n"] == 3
    assert result.error.startswith("HTTP 500:")
    assert result.text == ""


def test_api_runner_does_not_retry_a_usage_limit():
    """429 is the subscription window, not a transient fault.

    The run stage detects it and stops cleanly; retrying would only spend the
    remaining window faster.
    """
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, text="usage limit reached")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    result = runner.run("s", "u")
    assert attempts["n"] == 1
    assert result.error.startswith("HTTP 429:")


def test_api_runner_sends_no_ceiling_by_default():
    """No max_tokens at all, at any source length.

    The derived ceiling this replaces cost four chapters. Reasoning is emitted
    before visible content, so a budget that binds removes the translation
    rather than shortening it - every truncated call in 2026-08 recorded a full
    token spend and zero characters of English.
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    runner.run("s", "x" * 7400)
    assert "max_tokens" not in seen

    seen.clear()
    runner.run("s", "x" * 200_000)
    assert "max_tokens" not in seen


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
    # No ceiling is sent any more, so a truncation means the PROVIDER's own
    # default bound. The record has to say that rather than name a limit this
    # runner did not set - it is the only signal that an invisible cap exists.
    assert "provider default" in result.error
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


def test_api_runner_retries_a_cloudflare_origin_timeout():
    """524 fronts Command Code's API and means the provider produced nothing.

    Its body says so outright - "temporarily unavailable, please try again in
    a moment" - but it sat outside the 5xx tuple, so six anchor calls in a row
    were recorded as permanent failures.
    """
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(524, text='{"error":{"type":"server_error"}}')
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client, retry_backoff_s=0)
    result = runner.run("s", "u")
    assert attempts["n"] == 2
    assert result.error is None
    assert result.text == "ok"


def test_null_content_is_empty_text_not_a_crash():
    """A model can return content: null after spending its budget on reasoning.

    glm-5.3-flash does exactly that through Command Code. The None reached
    looks_like_refusal, raised TypeError out of runner.run, and would have
    aborted run_model along with every model still queued behind it.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 40000},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.text == ""
    assert result.refused is False
    assert result.tokens_out == 40000


def _capturing_runner(seen: list, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    return ApiRunner(
        endpoint="https://example.invalid/v1/chat/completions",
        model="vendor/model",
        api_key="k",
        temperature=0.3,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_extra_params_are_merged_into_the_payload():
    seen: list = []
    runner = _capturing_runner(
        seen, extra_params={"reasoning": {"effort": "high"}}
    )
    runner.run("sys", "src")
    assert seen[0]["reasoning"] == {"effort": "high"}


def test_no_extra_params_sends_the_payload_unchanged():
    """A model that asks for nothing must send what it sent before.

    This is the guarantee that adding the mechanism does not quietly change
    conditions for the ten models that do not use it.
    """
    seen: list = []
    _capturing_runner(seen).run("sys", "src")
    assert set(seen[0]) == {"model", "temperature", "messages"}


def test_extra_params_can_impose_a_ceiling_on_a_metered_route():
    """No ceiling is sent by default, so a metered route sets its own."""
    seen: list = []
    _capturing_runner(seen, extra_params={"max_tokens": 4096}).run("sys", "src")
    assert seen[0]["max_tokens"] == 4096


@pytest.mark.parametrize("stream", [False, True])
def test_unset_temperature_is_omitted_for_reasoning_models(stream):
    def handler(request):
        payload = json.loads(request.content)
        assert "temperature" not in payload
        assert payload["reasoning_effort"] == "max"
        assert payload["max_completion_tokens"] == 128000
        assert payload["messages"] == [
            {"role": "system", "content": "translate"},
            {"role": "user", "content": "source"},
        ]
        body = {
            "choices": [{"message": {"content": "translated"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        }
        if stream:
            body["choices"][0]["delta"] = body["choices"][0].pop("message")
            return httpx.Response(200, text=f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json=body)

    runner = ApiRunner(
        "https://api.openai.com/v1/chat/completions", "gpt-6-luna", "key", None,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        extra_params={"reasoning_effort": "max", "max_completion_tokens": 128000},
        stream=stream,
    )
    result = runner.run("translate", "source")
    assert result.text == "translated" and result.error is None
    assert (result.tokens_in, result.tokens_out, result.reasoning_tokens) == (20, 30, 10)
    assert result.route == "gpt-6-luna"


def test_completion_token_limit_is_reported_when_translation_is_truncated():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": "fragment"}, "finish_reason": "length"}]}
    )))
    runner = ApiRunner(
        "http://x", "m", "k", 0.3, client=client,
        extra_params={"max_completion_tokens": 128000},
    )
    result = runner.run("sys", "src")
    assert result.text == ""
    assert result.finish_reason == "length"
    assert "128000 tokens" in result.error


# --- free-to-paid fallback --------------------------------------------------


def _route_runner(responses, model="primary", fallback=None):
    """An ApiRunner whose transport replays `responses` in order."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ApiRunner(
        "http://x", model, "k", 0.3, client=client, attempts=1, fallback=fallback
    )


def _ok(text="translated"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_route_is_recorded_even_without_a_fallback():
    """Every record names the slug that answered, so a split is visible."""
    result = _route_runner([_ok()], model="only-route").run("s", "u")
    assert result.route == "only-route"


def test_falls_back_to_the_paid_route_when_free_will_not_serve():
    paid = _route_runner([_ok("paid answer")], model="paid")
    free = _route_runner(
        [httpx.Response(429, text="rate limited")], model="free", fallback=paid
    )
    result = free.run("s", "u")
    assert result.error is None
    assert result.text == "paid answer"
    # the record must name the route that actually produced the translation
    assert result.route == "paid"


def test_a_bad_translation_does_not_trigger_the_fallback():
    """A refusal is the model's own output and is identical on the paid tier.

    Spending money to receive the same answer twice is the failure mode this
    guards against.
    """
    paid = _route_runner([_ok("should never be reached")], model="paid")
    free = _route_runner([_ok("I cannot translate this")], model="free", fallback=paid)
    result = free.run("s", "u")
    assert result.refused is True
    assert result.route == "free"


def test_both_routes_failing_reports_both():
    paid = _route_runner([httpx.Response(500, text="paid down")], model="paid")
    free = _route_runner(
        [httpx.Response(429, text="free limited")], model="free", fallback=paid
    )
    result = free.run("s", "u")
    assert "free" in result.error and "paid" in result.error


def test_transient_429_is_retried_and_not_read_as_a_usage_limit():
    from ati.runstage import is_usage_limit

    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] < 2:
            return httpx.Response(
                429, text='{"message":"Upstream model provider is temporarily unavailable"}'
            )
        return _ok("recovered")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner(
        "http://x", "m", "k", 0.3, client=client, attempts=3, retry_backoff_s=0
    )
    result = runner.run("s", "u")
    assert result.text == "recovered"
    assert seen["n"] == 2
    assert not is_usage_limit(
        'HTTP 429: {"message":"Upstream model provider is temporarily unavailable"}'
    )


def _error_body_200(message="Upstream error from Chutes: overloaded"):
    return httpx.Response(200, json={"error": {"code": 502, "message": message}})


def test_an_error_object_in_a_200_response_is_reported_with_its_message():
    """OpenRouter's free routes answer 200 with an error in place of choices.
    The record used to say only "malformed response: 'choices'"."""
    result = _route_runner([_error_body_200()], model="free").run("s", "u")
    assert "provider error" in result.error
    assert "overloaded" in result.error
    assert result.text == ""


def test_an_error_object_in_a_200_response_triggers_the_fallback():
    paid = _route_runner([_ok("paid answer")], model="paid")
    free = _route_runner([_error_body_200()], model="free", fallback=paid)
    result = free.run("s", "u")
    assert result.error is None
    assert result.route == "paid"


# --- streaming --------------------------------------------------------------


def _sse(*chunks, done=True):
    lines = [": keep-alive", ""]
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines)


def _delta(text=None, finish=None):
    delta = {"content": text} if text is not None else {}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _stream_runner(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ApiRunner("http://x", "m", "k", 0.3, client=client, stream=True, **kwargs)


def test_a_streamed_answer_is_reassembled_with_usage():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        body = _sse(
            _delta("Hello, "),
            {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
            _delta("world."),
            _delta(finish="stop"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        )
        return httpx.Response(200, text=body)

    result = _stream_runner(handler).run("s", "u")
    assert result.error is None
    assert result.text == "Hello, world."
    assert result.tokens_out == 30
    assert result.reasoning_tokens == 20
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


def test_a_streamed_length_cutoff_is_still_a_truncation():
    def handler(request):
        return httpx.Response(200, text=_sse(_delta("partial"), _delta(finish="length")))

    result = _stream_runner(handler).run("s", "u")
    assert result.text == ""
    assert "truncated" in result.error


def test_a_stream_that_stops_early_is_not_scored():
    def handler(request):
        return httpx.Response(200, text=_sse(_delta("half a chap"), done=False))

    result = _stream_runner(handler, retry_backoff_s=0).run("s", "u")
    assert result.text == ""
    assert "stream ended" in result.error


def test_an_error_inside_the_stream_is_reported():
    def handler(request):
        return httpx.Response(
            200, text=_sse(_delta("some"), {"error": {"message": "upstream died"}})
        )

    result = _stream_runner(handler).run("s", "u")
    assert result.text == ""
    assert "upstream died" in result.error


def test_a_streamed_request_still_retries_a_524():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(524, text="Upstream model provider is temporarily unavailable")
        return httpx.Response(200, text=_sse(_delta("ok"), _delta(finish="stop")))

    result = _stream_runner(handler, retry_backoff_s=0).run("s", "u")
    assert result.error is None
    assert result.text == "ok"
    assert calls["n"] == 2


def test_streaming_is_off_unless_asked_for():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ApiRunner("http://x", "m", "k", 0.3, client=client).run("s", "u")
    assert "stream" not in seen


def test_a_dropped_connection_is_retried():
    """Command Code closed one stream mid-body on 2026-09-18 ("peer closed
    connection without sending complete message body")."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return httpx.Response(200, text=_sse(_delta("ok"), _delta(finish="stop")))

    result = _stream_runner(handler, retry_backoff_s=0).run("s", "u")
    assert result.error is None
    assert result.text == "ok"
    assert calls["n"] == 2


def test_a_stream_the_gateway_cut_short_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text=_sse(_delta("half"), done=False))
        return httpx.Response(200, text=_sse(_delta("whole"), _delta(finish="stop")))

    result = _stream_runner(handler, retry_backoff_s=0).run("s", "u")
    assert result.error is None
    assert result.text == "whole"
    assert calls["n"] == 2


def test_a_content_filter_finish_is_a_refusal_with_nothing_scored():
    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "partial"}, "finish_reason": "content_filter"}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = ApiRunner("http://x", "m", "k", 0.3, client=client).run("s", "u")
    assert result.refused is True
    assert result.text == ""
    assert result.finish_reason == "content_filter"


def test_a_provider_error_finish_is_a_route_failure_and_falls_back():
    """nemotron-3.5-lightning's free route stopped generations at 600s."""
    paid = _route_runner([_ok("paid answer")], model="paid")
    free = _route_runner(
        [httpx.Response(200, json={"choices": [{"message": {"content": "half"}, "finish_reason": "error"}]})],
        model="free",
        fallback=paid,
    )
    result = free.run("s", "u")
    assert result.error is None
    assert result.text == "paid answer"
    assert result.route == "paid"


def test_the_finish_reason_survives_streaming():
    def handler(request):
        return httpx.Response(200, text=_sse(_delta("done"), _delta(finish="stop")))

    result = _stream_runner(handler).run("s", "u")
    assert result.finish_reason == "stop"


# --- total deadline on a streamed call ---------------------------------------


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Trickle(httpx.SyncByteStream):
    """A stream that keeps sending, advancing a fake clock per chunk."""

    def __init__(self, chunks, clock, step_s):
        self._chunks, self._clock, self._step = chunks, clock, step_s

    def __iter__(self):
        for chunk in self._chunks:
            self._clock.now += self._step
            yield chunk.encode("utf-8")


def _trickling_runner(clock, chunks, step_s, calls, **kwargs):
    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_Trickle(chunks, clock, step_s),
        )

    return _stream_runner(handler, clock_fn=clock, retry_backoff_s=0, **kwargs)


def test_a_stream_that_never_ends_is_cut_at_the_total_deadline():
    """httpx's timeout is per read, so keep-alives alone could hold a call open
    forever. The total deadline ends it, and it is not retried."""
    clock, calls = _Clock(), []
    keepalives = [": keep-alive\n\n"] * 100
    runner = _trickling_runner(clock, keepalives, 60.0, calls, deadline_s=600.0)
    result = runner.run("s", "u")
    assert result.text == ""
    assert "timed out" in result.error and "deadline" in result.error
    assert len(calls) == 1
    assert clock.now <= 660.0  # stopped reading as soon as it ran over


def test_a_long_stream_inside_the_deadline_is_kept():
    clock, calls = _Clock(), []
    body = _sse(_delta("slow "), _delta("but whole"), _delta(finish="stop"))
    chunks = body.splitlines(keepends=True)
    runner = _trickling_runner(clock, chunks, 120.0, calls, deadline_s=3600.0)
    result = runner.run("s", "u")
    assert result.error is None
    assert result.text == "slow but whole"
    assert clock.now > 900.0  # longer than the per-read timeout itself


def test_a_deadline_overrun_counts_as_a_route_failure_for_the_fallback():
    clock, calls = _Clock(), []
    paid = _route_runner([_ok("paid answer")], model="paid")
    free = _trickling_runner(
        clock, [": keep-alive\n\n"] * 100, 60.0, calls, deadline_s=600.0, fallback=paid
    )
    result = free.run("s", "u")
    assert result.error is None
    assert result.route == "paid"


def test_the_runner_defaults_come_from_the_one_timeout_constant():
    from ati.config import DEFAULT_TIMEOUT_S
    from ati.runners.agy import AgyRunner
    from ati.runners.cli import CliRunner

    assert AgyRunner("m")._timeout_s == DEFAULT_TIMEOUT_S
    assert CliRunner(["x"])._timeout_s == DEFAULT_TIMEOUT_S
    runner = ApiRunner("http://x", "m", "k", 0.3)
    assert runner._client.timeout.read == DEFAULT_TIMEOUT_S
    assert runner._client.timeout.connect == 60.0


def test_an_empty_system_prompt_is_omitted_not_sent_empty():
    """The judge stages pass ""; xAI rejects an empty system message with
    HTTP 400 "system message must have content"."""
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ApiRunner("http://x", "m", "k", 0.0, client=client).run("", "judge this")
    assert [m["role"] for m in seen["messages"]] == ["user"]


def test_a_real_system_prompt_is_still_sent():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ApiRunner("http://x", "m", "k", 0.0, client=client).run("SYSTEM", "source")
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert seen["messages"][0]["content"] == "SYSTEM"


def test_api_runner_sends_declared_headers_and_a_fresh_session_per_call():
    """opencode Go refuses a request without x-opencode-session. Each chapter
    is its own conversation, so each call gets its own id."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    runner = ApiRunner(
        endpoint="https://example.invalid/v1/chat/completions",
        model="vendor/model",
        api_key="k",
        temperature=0.3,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        headers={"User-Agent": "ati-translation-index/1.0"},
        session_header="x-opencode-session",
    )
    runner.run("sys", "src")
    runner.run("sys", "src")
    assert all(h["user-agent"] == "ati-translation-index/1.0" for h in seen)
    sessions = [h["x-opencode-session"] for h in seen]
    assert all(sessions) and sessions[0] != sessions[1]
    assert all(h["authorization"] == "Bearer k" for h in seen)


def test_api_runner_declared_headers_cannot_replace_the_credential():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    runner = ApiRunner(
        endpoint="https://example.invalid/v1/chat/completions",
        model="vendor/model",
        api_key="k",
        temperature=0.3,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        headers={"Authorization": "Bearer other"},
    )
    runner.run("sys", "src")
    assert seen[0]["authorization"] == "Bearer k"
