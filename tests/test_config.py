import yaml
import os
from pathlib import Path

import pytest

from ati.config import (
    AXES,
    ConfigError,
    load_env_file,
    load_models,
    load_weights,
)


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_axes_are_the_six_rubric_axes():
    assert AXES == (
        "accuracy",
        "terminology",
        "register",
        "fluency",
        "fidelity",
        "sanitization",
    )


def test_load_models_reads_api_and_cli_entries(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        """
- id: openrouter/some-model
  runner: api
  endpoint: https://openrouter.ai/api/v1/chat/completions
  model: vendor/some-model
- id: claude-code
  runner: cli
  command: ["claude", "-p"]
  temperature: 0.0
""",
    )
    models = load_models(p)
    assert [m.id for m in models] == ["openrouter/some-model", "claude-code"]
    assert models[0].runner == "api"
    assert models[0].temperature == 0.3  # default
    assert models[1].command == ["claude", "-p"]
    assert models[1].temperature == 0.0
    assert models[0].anchor is None


def test_load_models_rejects_duplicate_ids(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        """
- id: dupe
  runner: cli
  command: ["x"]
- id: dupe
  runner: cli
  command: ["y"]
""",
    )
    with pytest.raises(ConfigError, match="duplicate model id"):
        load_models(p)


def test_load_models_rejects_unknown_runner(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: telepathy\n")
    with pytest.raises(ConfigError, match="unknown runner"):
        load_models(p)


def test_load_models_rejects_api_entry_without_model(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: api\n  endpoint: http://x\n")
    with pytest.raises(ConfigError, match="requires 'model'"):
        load_models(p)


def test_load_models_rejects_cli_entry_without_command(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: cli\n")
    with pytest.raises(ConfigError, match="requires 'command'"):
        load_models(p)


def test_load_models_accepts_anchor_labels(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        """
- id: strong
  runner: cli
  command: ["x"]
  anchor: strong
- id: weak
  runner: cli
  command: ["y"]
  anchor: weak
""",
    )
    models = load_models(p)
    assert [m.anchor for m in models] == ["strong", "weak"]


def test_load_models_rejects_bad_anchor_label(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        "- id: m\n  runner: cli\n  command: [\"x\"]\n  anchor: medium\n",
    )
    with pytest.raises(ConfigError, match="anchor must be"):
        load_models(p)


def test_load_weights_returns_all_axes(tmp_path):
    p = write(
        tmp_path,
        "weights.yaml",
        """
accuracy: 0.25
terminology: 0.20
register: 0.15
fluency: 0.15
fidelity: 0.15
sanitization: 0.10
""",
    )
    w = load_weights(p)
    assert set(w) == set(AXES)
    assert w["accuracy"] == 0.25


def test_load_weights_rejects_sum_not_one(tmp_path):
    p = write(
        tmp_path,
        "weights.yaml",
        """
accuracy: 0.5
terminology: 0.5
register: 0.5
fluency: 0.15
fidelity: 0.15
sanitization: 0.10
""",
    )
    with pytest.raises(ConfigError, match="must sum to 1.0"):
        load_weights(p)


def test_load_weights_rejects_missing_axis(tmp_path):
    p = write(tmp_path, "weights.yaml", "accuracy: 1.0\n")
    with pytest.raises(ConfigError, match="missing axis"):
        load_weights(p)


def test_model_timeout_defaults_above_measured_chapter_time(tmp_path):
    """Sized from the output ceiling, not from the typical chapter.

    A normal chapter returns in 190-420s, but the timeout has to cover the
    slowest call the token budget still permits: 49,152 tokens at the 60
    tokens/sec floor measured on this corpus is ~820s.
    """
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: cli\n  command: [\"x\"]\n")
    assert load_models(p)[0].timeout_s == 900.0


def test_model_timeout_can_be_overridden(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        "- id: m\n  runner: cli\n  command: [\"x\"]\n  timeout_s: 60\n",
    )
    assert load_models(p)[0].timeout_s == 60.0


def test_api_key_env_defaults_to_openrouter(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: api\n  model: x\n")
    assert load_models(p)[0].api_key_env == "OPENROUTER_API_KEY"


def test_api_key_env_is_per_model(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        "- id: m\n  runner: api\n  model: x\n  api_key_env: OPENCODE_API_KEY\n",
    )
    assert load_models(p)[0].api_key_env == "OPENCODE_API_KEY"


def test_params_must_be_a_mapping(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "- id: m\n  runner: api\n  model: v/m\n  params: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_models(path)


def test_params_may_not_override_structural_fields(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "- id: m\n  runner: api\n  model: v/m\n  params:\n    messages: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_models(path)


def test_params_are_loaded_when_valid(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "- id: m\n  runner: api\n  model: v/m\n"
        "  params:\n    reasoning:\n      effort: high\n",
        encoding="utf-8",
    )
    assert load_models(path)[0].params == {"reasoning": {"effort": "high"}}


def test_load_env_file_sets_names_and_reports_them(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "OPENROUTER_API_KEY=or-value\n"
        "export OPENCODE_API_KEY='oc-value'\n"
        'COMMANDCODE_API_KEY="cc-value"\n',
        encoding="utf-8",
    )
    for name in ("OPENROUTER_API_KEY", "OPENCODE_API_KEY", "COMMANDCODE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    loaded = load_env_file(env)

    assert sorted(loaded) == [
        "COMMANDCODE_API_KEY",
        "OPENCODE_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    assert os.environ["OPENCODE_API_KEY"] == "oc-value"
    assert os.environ["COMMANDCODE_API_KEY"] == "cc-value"


def test_load_env_file_does_not_override_an_exported_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")

    assert load_env_file(env) == []
    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_load_env_file_skips_empty_placeholders(tmp_path, monkeypatch):
    """An unfilled placeholder must not shadow a real key with ''."""
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert load_env_file(env) == []
    assert "OPENROUTER_API_KEY" not in os.environ


def test_load_env_file_is_silent_when_absent(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


# --- Gemini is reached only through Antigravity -----------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "- id: g\n  runner: api\n  model: google/gemini-3.7-flash\n",
        "- id: gemini-judge\n  runner: api\n  model: something\n",
        "- id: g\n  runner: cli\n  command: [opencode, run, -m, google/gemini-3.8-flash]\n",
        "- id: g\n  runner: api\n  model: x/free\n  fallback:\n    model: google/gemini-3.7-flash\n",
    ],
)
def test_gemini_is_refused_on_any_route_but_antigravity(tmp_path, entry):
    with pytest.raises(ConfigError, match="Antigravity"):
        load_models(write(tmp_path, "m.yaml", entry))


def test_gemini_through_antigravity_is_accepted(tmp_path):
    p = write(
        tmp_path, "m.yaml", "- id: gemini-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n"
    )
    assert load_models(p)[0].runner == "agy"


def test_stream_defaults_off_and_is_read_when_set(tmp_path):
    p = write(
        tmp_path,
        "m.yaml",
        "- id: a\n  runner: api\n  model: x\n- id: b\n  runner: api\n  model: y\n  stream: true\n",
    )
    a, b = load_models(p)
    assert a.stream is False
    assert b.stream is True


# --- roles, ids and the real configuration ------------------------------------

REPO = Path(__file__).resolve().parents[1]


def test_judges_and_antigravity_rows_are_not_translation_subjects(tmp_path):
    from ati.config import subjects

    p = write(
        tmp_path,
        "m.yaml",
        "- id: a\n  runner: api\n  model: x\n"
        "- id: j\n  role: judge\n  runner: api\n  model: y\n"
        "- id: g\n  runner: agy\n  model: gemini-x\n",
    )
    assert [m.id for m in subjects(load_models(p))] == ["a"]


def test_an_unknown_role_is_refused(tmp_path):
    p = write(tmp_path, "m.yaml", "- id: a\n  role: referee\n  runner: api\n  model: x\n")
    with pytest.raises(ConfigError, match="role"):
        load_models(p)


@pytest.mark.parametrize("bad", ["a:b", "a?b", "a*b", "a|b", 'a"b'])
def test_an_id_that_cannot_be_a_windows_file_name_is_refused(tmp_path, bad):
    p = write(tmp_path, "m.yaml", f"- id: '{bad}'\n  runner: api\n  model: x\n")
    with pytest.raises(ConfigError, match="Windows"):
        load_models(p)


def test_unknown_keys_in_ati_yaml_are_refused(tmp_path):
    from ati.config import load_defaults

    p = write(tmp_path, "ati.yaml", "judge-model: x\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_defaults(p)


def test_the_real_roster_loads_and_its_judges_are_not_subjects():
    """`ati run` with no --model would otherwise translate through the judges -
    gemini-judge through Antigravity on the operator's own account."""
    from ati.config import load_defaults, subjects

    roster = load_models(REPO / "config" / "models.yaml")
    ids = {m.id for m in subjects(roster)}
    assert "gemini-judge" not in ids and "sonnet-judge" not in ids
    # Every judge row is excluded, and so is every agy row: reaching Gemini
    # costs the operator's own Antigravity quota, so an agy subject runs only
    # when it is named with --model. Asserted as the rule rather than a count,
    # which a new roster row would break for no reason.
    assert ids == {
        m.id for m in roster if m.role != "judge" and m.runner != "agy"
    }
    assert any(m.runner == "agy" and m.role != "judge" for m in roster)
    assert len(ids) >= 20
    defaults = load_defaults(REPO / "config" / "ati.yaml")
    judge = next(m for m in roster if m.id == defaults["judge_model"])
    assert judge.runner == "agy" and judge.role == "judge"


def test_system_prompt_flag_reaches_the_runner(tmp_path):
    """Declared in the roster and carried onto the runner.

    The field was added to ModelConfig and read by CliRunner before the loader
    passed it, so a row that asked for `--system-prompt` silently fell back to
    stdin delivery - which runs the translation underneath the coding agent's
    own system prompt instead of replacing it.
    """
    path = tmp_path / "models.yaml"
    path.write_text(
        "- id: m\n"
        "  runner: cli\n"
        "  system_prompt_flag: --system-prompt\n"
        "  command: [echo, hi]\n",
        encoding="utf-8",
    )
    (model,) = load_models(path)
    assert model.system_prompt_flag == "--system-prompt"


def test_every_roster_row_declaring_a_field_has_it_loaded(tmp_path):
    """Every key the shipped roster uses maps onto a ModelConfig field."""
    import dataclasses

    from ati.config import ModelConfig

    known = {f.name for f in dataclasses.fields(ModelConfig)}
    raw = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))
    used = {key for entry in raw for key in entry}
    assert used <= known, f"roster keys with no ModelConfig field: {used - known}"


def test_headers_session_header_and_price_are_loaded(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "- id: m\n"
        "  runner: api\n"
        "  model: vendor/m\n"
        "  headers: {User-Agent: ati-translation-index/1.0}\n"
        "  session_header: x-opencode-session\n"
        "  price: {input: 3.0, output: 15.0}\n",
        encoding="utf-8",
    )
    (model,) = load_models(path)
    assert model.headers == {"User-Agent": "ati-translation-index/1.0"}
    assert model.session_header == "x-opencode-session"
    assert model.price == {"input": 3.0, "output": 15.0}


def test_a_price_needs_both_input_and_output(tmp_path):
    import pytest

    path = tmp_path / "models.yaml"
    path.write_text(
        "- id: m\n  runner: api\n  model: vendor/m\n  price: {output: 15.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_models(path)
