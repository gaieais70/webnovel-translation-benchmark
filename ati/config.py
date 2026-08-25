"""Loading and validation of models.yaml and weights.yaml."""

from __future__ import annotations

from dataclasses import dataclass
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

VALID_RUNNERS = ("api", "cli", "manual")
VALID_ANCHORS = ("strong", "weak")
DEFAULT_TEMPERATURE = 0.3


class ConfigError(Exception):
    """Raised when a config file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    runner: str
    model: str | None = None
    endpoint: str | None = None
    command: list[str] | None = None
    temperature: float = DEFAULT_TEMPERATURE
    anchor: str | None = None


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
