"""Loading and validation of models.yaml and weights.yaml."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml

AXES: tuple[str, ...] = (
    "accuracy",
    "terminology",
    "register",
    "fluency",
    "fidelity",
    "sanitization",
)

VALID_RUNNERS = ("api", "cli", "manual", "agy")
VALID_ANCHORS = ("strong", "weak")
DEFAULT_TEMPERATURE = 0.3
# Measured across models: 190-240s per chapter is normal. Safe to be generous
# now that max_tokens bounds how much a model can generate in one call.
# Raised from 600 once the output budget grew. Measured throughput on this
# corpus is 60-72 completion tokens/sec, so a call allowed to reach the
# 49,152-token clamp needs up to ~820s. At 600 the budget increase would have
# converted truncations into read timeouts, which are deliberately not
# retried, and the run would have failed the same models for a new reason.
DEFAULT_TIMEOUT_S = 900.0
# Each provider needs its own credential. Defaulting every model to one env
# var means a mixed pool sends the wrong key to one of them and gets 401s.
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


def load_env_file(path: Path | str = ".env") -> list[str]:
    """Read KEY=VALUE lines from .env into the environment.

    .gitignore has listed .env since the project started, but nothing ever
    read it, so provider keys had to be exported by hand every session and a
    missing one surfaced as a 401 from the provider rather than as a setup
    error. Deliberately stdlib: the two runtime dependencies are httpx and
    PyYAML, and a dotenv dependency is not worth twelve lines.

    A value already in the environment wins, so an explicit export still
    overrides the file. Returns the names it set - never the values, which
    must not reach a log.
    """
    file = Path(path)
    if not file.is_file():
        return []
    loaded: list[str] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        # An empty value is a placeholder, not a credential. Setting it would
        # shadow a real exported key with the empty string.
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


class ConfigError(Exception):
    """Raised when a config file is malformed or internally inconsistent."""


def requires_antigravity(*names: str | None) -> bool:
    """Whether any of these ids or slugs names a Gemini model.

    The operator's rule, and a hard one: Gemini is reached only through the
    Antigravity CLI on their own account. Command Code and OpenRouter both
    list Gemini slugs, so a single mistyped roster row would otherwise send it
    through a gateway instead, with nothing in the output to show it.
    """
    return any("gemini" in (name or "").lower() for name in names)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    runner: str
    model: str | None = None
    endpoint: str | None = None
    command: list[str] | None = None
    temperature: float = DEFAULT_TEMPERATURE
    anchor: str | None = None
    json_schema: str | None = None
    sandbox: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    api_key_env: str = DEFAULT_API_KEY_ENV
    # Extra request fields merged into the API payload verbatim. Exists
    # because reasoning effort is spelled differently by every provider -
    # OpenRouter nests it under `reasoning`, OpenAI-compatible routes take a
    # flat `reasoning_effort` - and hardcoding one spelling would send a field
    # the other two silently ignore. Opt-in: a model with no `params` sends
    # exactly the payload it sent before this existed, so adding it changes
    # conditions for no model that does not ask for it.
    params: dict | None = None
    # An alternate route to try when this one fails to serve at all. Same
    # shape as a roster entry: `model`, and optionally `endpoint`,
    # `api_key_env` and `params`, each defaulting to this entry's value.
    # Exists so a free tier can be tried first and a paid one used only when
    # the free route will not answer, which is a routing decision rather than
    # a second model - `route` on each run record says which one served it.
    fallback: dict | None = None
    # Stream the response (server-sent events) instead of waiting for one
    # buffered body. Needed on gateways that drop long-running buffered
    # requests; see ApiRunner. Transport only - the model generates the same.
    stream: bool = False


def load_models(path: Path) -> list[ModelConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ConfigError("models.yaml must contain a list of model entries")

    models: list[ModelConfig] = []
    seen: set[str] = set()
    for entry in raw:
        model_id = entry.get("id")
        if not model_id:
            raise ConfigError("every model entry requires 'id'")
        if model_id in seen:
            raise ConfigError(f"duplicate model id: {model_id}")
        seen.add(model_id)

        runner = entry.get("runner")
        if runner not in VALID_RUNNERS:
            raise ConfigError(f"unknown runner {runner!r} for model {model_id}")
        if runner == "api" and not entry.get("model"):
            raise ConfigError(f"model {model_id}: runner 'api' requires 'model'")
        if runner == "cli" and not entry.get("command"):
            raise ConfigError(f"model {model_id}: runner 'cli' requires 'command'")
        if runner == "agy" and not entry.get("model"):
            raise ConfigError(f"model {model_id}: runner 'agy' requires 'model'")
        fallback_entry = entry.get("fallback")
        fallback_slug = (
            fallback_entry.get("model") if isinstance(fallback_entry, dict) else None
        )
        if runner != "agy" and requires_antigravity(
            model_id,
            entry.get("model"),
            fallback_slug,
            " ".join(entry.get("command") or []),
        ):
            raise ConfigError(
                f"model {model_id}: Gemini may only be reached through "
                f"Antigravity (runner: agy), not runner {runner!r}"
            )

        params = entry.get("params")
        if params is not None:
            if not isinstance(params, dict):
                raise ConfigError(
                    f"model {model_id}: 'params' must be a mapping"
                )
            # These are structural: the runner builds them from the passage
            # and the roster. Letting a params block override them would mean
            # a run silently translating something other than its own source.
            for reserved in ("model", "messages"):
                if reserved in params:
                    raise ConfigError(
                        f"model {model_id}: 'params' may not override {reserved!r}"
                    )

        fallback = entry.get("fallback")
        if fallback is not None:
            if not isinstance(fallback, dict):
                raise ConfigError(
                    f"model {model_id}: 'fallback' must be a mapping"
                )
            if not fallback.get("model"):
                raise ConfigError(
                    f"model {model_id}: 'fallback' requires 'model'"
                )
            # A fallback that resolves to the same slug on the same endpoint
            # is not a fallback; it would just repeat a call that already
            # failed and make the record claim a second route was tried.
            same_endpoint = fallback.get("endpoint", entry.get("endpoint"))
            if (
                fallback["model"] == entry.get("model")
                and same_endpoint == entry.get("endpoint")
            ):
                raise ConfigError(
                    f"model {model_id}: 'fallback' is the same route as the primary"
                )

        anchor = entry.get("anchor")
        if anchor is not None and anchor not in VALID_ANCHORS:
            raise ConfigError(
                f"model {model_id}: anchor must be 'strong' or 'weak', got {anchor!r}"
            )

        models.append(
            ModelConfig(
                id=model_id,
                runner=runner,
                model=entry.get("model"),
                endpoint=entry.get("endpoint"),
                command=entry.get("command"),
                temperature=float(entry.get("temperature", DEFAULT_TEMPERATURE)),
                anchor=anchor,
                json_schema=entry.get("json_schema"),
                sandbox=bool(entry.get("sandbox", True)),
                timeout_s=float(entry.get("timeout_s", DEFAULT_TIMEOUT_S)),
                api_key_env=entry.get("api_key_env", DEFAULT_API_KEY_ENV),
                params=params,
                fallback=fallback,
                stream=bool(entry.get("stream", False)),
            )
        )
    return models


def load_weights(path: Path) -> dict[str, float]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("weights.yaml must contain a mapping of axis to weight")

    for axis in AXES:
        if axis not in raw:
            raise ConfigError(f"weights.yaml missing axis: {axis}")
    extra = set(raw) - set(AXES)
    if extra:
        raise ConfigError(f"weights.yaml has unknown axes: {sorted(extra)}")

    weights = {axis: float(raw[axis]) for axis in AXES}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"weights must sum to 1.0, got {total}")
    return weights


def load_defaults(path: Path) -> dict:
    """Operator defaults from config/ati.yaml. Absent file means no defaults.

    Precedence is: explicit CLI flag, then this file, then the built-in
    default baked into the CLI.
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("ati.yaml must contain a mapping")
    return raw
