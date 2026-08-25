from ati.runners.manual import ManualRunner


def test_manual_runner_writes_prompt_when_no_response_exists(tmp_path):
    runner = ManualRunner(root=tmp_path, model_id="web-model")
    result = runner.run("SYSTEM", "SOURCE")
    prompt_file = tmp_path / "web-model" / "pending.prompt"
    assert prompt_file.exists()
    assert "SOURCE" in prompt_file.read_text(encoding="utf-8")
    assert result.error == "awaiting manual response"
    assert result.text == ""


def test_manual_runner_reads_pasted_response(tmp_path):
    runner = ManualRunner(root=tmp_path, model_id="web-model")
    runner.set_task_id("cn-01.0")
    (tmp_path / "web-model").mkdir(parents=True, exist_ok=True)
    (tmp_path / "web-model" / "cn-01.0.response").write_text(
        "pasted translation", encoding="utf-8"
    )
    result = runner.run("SYSTEM", "SOURCE")
    assert result.text == "pasted translation"
    assert result.error is None
    assert result.cost_usd is None
