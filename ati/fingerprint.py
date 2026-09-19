"""Fingerprints: what produced a judgment, a comparison, or a published result.

A score is only meaningful alongside the judge that gave it and the words
that judge was given. Before this, a judgment recorded only the judge's id,
and nothing checked even that: switching `--judge-model` and rerunning
`ati judge` kept every old score while the report named the new judge, and
editing the judge prompt invalidated nothing - on run 2026-09, 35 judgments
made before the no-tools line had to be moved aside by hand.

A judge fingerprint hashes everything that changes what the judge sees or how
it answers: its id, the route behind the id (runner, model slug, endpoint and
sampling settings, and for Antigravity the no-tools preamble and the output
schema), and the prompt frame - the stage's template with the per-item text
replaced by placeholders, so a change to either the template file or the code
that wraps it is caught.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ati.config import ModelConfig


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str | None:
    """Hex digest of a file's bytes, or None if it cannot be read."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _read_text(path: Path | str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def judge_route(
    judge_id: str,
    model: ModelConfig | None,
    default_endpoint: str,
    json_schema: str | None = None,
) -> dict:
    """The settings behind a judge id that change the answers it gives.

    Timeouts and gate spacing are left out on purpose: they change whether an
    answer arrives, not what it says.
    """
    if model is None:
        # An undeclared judge: a plain API call at temperature 0.
        return {
            "runner": "api",
            "model": judge_id,
            "endpoint": default_endpoint,
            "temperature": 0.0,
        }
    route: dict = {"runner": model.runner, "model": model.model or model.id}
    if model.runner == "api":
        route.update(
            endpoint=model.endpoint or default_endpoint,
            temperature=model.temperature,
            params=model.params,
        )
    elif model.runner == "agy":
        # Imported here so this module does not pull in subprocess plumbing
        # for callers that only hash files.
        from ati.runners.agy import NO_TOOLS_PREAMBLE

        schema = json_schema if json_schema is not None else model.json_schema
        route.update(
            sandbox=model.sandbox,
            preamble=NO_TOOLS_PREAMBLE,
            schema=_read_text(schema),
        )
    elif model.runner == "cli":
        route.update(command=model.command)
    return route


def judge_fingerprint(judge_id: str, route: dict, prompt_frame: str) -> str:
    """A short, stable hash of the judge, its route, and its prompt frame."""
    payload = json.dumps(
        {"judge_id": judge_id, "route": route, "prompt": prompt_frame},
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256_text(payload)[:16]


def rubric_prompt_frame(template: str) -> str:
    """The rubric prompt with every per-item slot replaced by a placeholder."""
    from ati.judge import build_judge_prompt

    return "\n---\n".join(
        build_judge_prompt(
            template=template,
            source_text="{source}",
            translation="{translation}",
            reference=None,
            expect_term_table=term_table,
        )
        for term_table in (False, True)
    )


def pairwise_prompt_frame(template: str) -> str:
    """The pairwise prompt with both translations and the source as placeholders."""
    from ati.pairwise import build_pair_prompt

    return build_pair_prompt(template, "{source}", "{translation_a}", "{translation_b}")


def input_fingerprints(paths: dict[str, Path | str]) -> dict[str, str | None]:
    """sha256 of each named input file; None marks one that could not be read."""
    return {name: sha256_file(path) for name, path in paths.items()}


def code_fingerprint(package_dir: Path | str | None = None) -> str:
    """A hash of the harness's own source, independent of git and line endings.

    The same checkout gives the same value on Windows and POSIX, so a result
    can be tied to the exact code that aggregated it.
    """
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]
