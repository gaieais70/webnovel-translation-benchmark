import json
import os

import pytest

from ati import fileio
from ati.fileio import iter_records, read_json, write_json_atomic, write_text_atomic


def test_atomic_write_creates_parents_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "a" / "b" / "rec.json"
    write_json_atomic(target, {"model_id": "m", "text": "한국어"})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "model_id": "m",
        "text": "한국어",
    }
    assert [p.name for p in target.parent.iterdir()] == ["rec.json"]


def test_atomic_write_matches_the_plain_write_it_replaced(tmp_path):
    """Byte-identical to Path.write_text, so nothing downstream sees a change."""
    payload = {"text": "line one\nline two", "n": 1}
    old = tmp_path / "old.json"
    old.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    new = tmp_path / "new.json"
    write_json_atomic(new, payload)
    assert new.read_bytes() == old.read_bytes()


def test_atomic_write_replaces_an_existing_file(tmp_path):
    target = tmp_path / "rec.json"
    write_text_atomic(target, "first")
    write_text_atomic(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_a_reader_holding_the_file_open_only_delays_the_replace(tmp_path, monkeypatch):
    """Windows refuses os.replace onto a file another process is reading."""
    target = tmp_path / "rec.json"
    real_replace = os.replace
    failures = {"left": 2}

    def flaky_replace(src, dst):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError("in use")
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", flaky_replace)
    slept = []
    write_text_atomic(target, "done", sleep_fn=slept.append)
    assert target.read_text(encoding="utf-8") == "done"
    assert len(slept) == 2


def test_a_failed_write_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "rec.json"
    write_text_atomic(target, "original")

    def always_busy(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(fileio.os, "replace", always_busy)
    with pytest.raises(PermissionError):
        write_text_atomic(target, "new", sleep_fn=lambda s: None)
    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["rec.json"]


@pytest.mark.parametrize("content", ["", '{"model_id": "m", "te', "[1, 2]", "\x00\x01"])
def test_read_json_returns_none_for_anything_unusable(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    assert read_json(path) is None


def test_read_json_returns_none_for_a_missing_file(tmp_path):
    assert read_json(tmp_path / "absent.json") is None


def test_iter_records_skips_and_reports_bad_files(tmp_path):
    good = tmp_path / "good.json"
    good.write_text('{"model_id": "m"}', encoding="utf-8")
    torn = tmp_path / "torn.json"
    torn.write_text('{"model_id": "m", "te', encoding="utf-8")
    stray = tmp_path / "stray.json"
    stray.write_text('{"something": "else"}', encoding="utf-8")
    warnings = []
    records = list(
        iter_records([good, torn, stray], required=("model_id",), warn=warnings.append)
    )
    assert records == [(good, {"model_id": "m"})]
    assert len(warnings) == 2
