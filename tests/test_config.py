from pathlib import Path

import pytest

from ati.config import AXES, ConfigError, load_models, load_weights


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
