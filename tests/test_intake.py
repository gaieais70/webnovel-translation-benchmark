import json

import pytest
import yaml

from ati.corpus import passage_path, sha256_text
from ati.intake import (
    DEFAULT_CHARS,
    IntakeError,
    add_files,
    make_id,
    normalize,
)


def write_raw(tmp_path, name, text):
    d = tmp_path / "raw"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def call(tmp_path, files, kind="passage", work="skyfall", chars=DEFAULT_CHARS):
    return add_files(
        files=files,
        work=work,
        lang="zh",
        genre="xianxia",
        kind=kind,
        url="https://example.invalid/skyfall",
        corpus_root=tmp_path / "corpus",
        manifest_path=tmp_path / "corpus" / "manifest.json",
        tasks_path=tmp_path / "config" / "tasks.yaml",
        chars=chars,
    )


def read_manifest(tmp_path):
    return json.loads(
        (tmp_path / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )


def read_tasks(tmp_path):
    return yaml.safe_load(
        (tmp_path / "config" / "tasks.yaml").read_text(encoding="utf-8")
    )


def test_normalize_strips_bom_and_crlf():
    assert normalize("﻿line one\r\nline two") == "line one\nline two"


def test_make_id_numbers_passages_and_chapters_differently():
    assert make_id("skyfall", "passage", 1) == "skyfall-01"
    assert make_id("skyfall", "refusal", 12) == "skyfall-12"
    assert make_id("skyfall", "consistency", 1) == "skyfall-ch01"


def test_add_files_writes_passage_text_and_manifest_entry(tmp_path):
    f = write_raw(tmp_path, "a.txt", "云澈抬起头")
    result = call(tmp_path, [f])

    assert result.added == ["skyfall-01"]
    assert result.unchanged == []

    stored = passage_path(tmp_path / "corpus", "skyfall-01")
    assert stored.read_text(encoding="utf-8") == "云澈抬起头"

    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["id"] == "skyfall-01"
    assert entry["lang"] == "zh"
    assert entry["genre"] == "xianxia"
    assert entry["sha256"] == sha256_text("云澈抬起头")
    assert entry["start"] == 0
    assert entry["end"] == 5
    assert entry["has_reference"] is False
    assert entry["source_file"] == "a.txt"


def test_add_files_slices_passages_to_chars(tmp_path):
    f = write_raw(tmp_path, "a.txt", "x" * 2000)
    call(tmp_path, [f], chars=800)
    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["end"] == 800
    stored = passage_path(tmp_path / "corpus", "skyfall-01")
    assert len(stored.read_text(encoding="utf-8")) == 800
    assert entry["sha256"] == sha256_text("x" * 800)


def test_add_files_keeps_whole_chapter_for_consistency(tmp_path):
    f = write_raw(tmp_path, "ch1.txt", "y" * 2000)
    call(tmp_path, [f], kind="consistency")
    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["id"] == "skyfall-ch01"
    assert entry["end"] == 2000
    assert entry["series"] == "skyfall"
    assert entry["series_index"] == 1
    assert entry["chapter"] == "1"


def test_add_files_is_idempotent_for_identical_text(tmp_path):
    f = write_raw(tmp_path, "a.txt", "same text")
    call(tmp_path, [f])
    result = call(tmp_path, [f])
    assert result.added == []
    assert result.unchanged == ["skyfall-01"]
    assert len(read_manifest(tmp_path)["passages"]) == 1


def test_add_files_raises_when_text_changed_under_existing_id(tmp_path):
    f = write_raw(tmp_path, "a.txt", "original")
    call(tmp_path, [f])
    f.write_text("different", encoding="utf-8")
    with pytest.raises(IntakeError, match="skyfall-01"):
        call(tmp_path, [f])


def test_add_files_continues_numbering_on_later_calls(tmp_path):
    a = write_raw(tmp_path, "a.txt", "first")
    call(tmp_path, [a])
    b = write_raw(tmp_path, "b.txt", "second")
    result = call(tmp_path, [b])
    assert result.added == ["skyfall-02"]
    assert [e["id"] for e in read_manifest(tmp_path)["passages"]] == [
        "skyfall-01",
        "skyfall-02",
    ]


def test_add_files_numbers_multiple_files_in_sorted_order(tmp_path):
    c = write_raw(tmp_path, "ch03.txt", "three")
    a = write_raw(tmp_path, "ch01.txt", "one")
    b = write_raw(tmp_path, "ch02.txt", "two")
    result = call(tmp_path, [c, a, b], kind="consistency")
    assert result.added == ["skyfall-ch01", "skyfall-ch02", "skyfall-ch03"]
    stored = passage_path(tmp_path / "corpus", "skyfall-ch01")
    assert stored.read_text(encoding="utf-8") == "one"


def test_add_files_writes_task_entry(tmp_path):
    f = write_raw(tmp_path, "a.txt", "text")
    result = call(tmp_path, [f])
    assert result.task_id == "skyfall-passage"
    tasks = read_tasks(tmp_path)
    assert tasks == [
        {"id": "skyfall-passage", "type": "passage", "passages": ["skyfall-01"]}
    ]


def test_add_files_extends_existing_task_without_duplicates(tmp_path):
    a = write_raw(tmp_path, "a.txt", "one")
    call(tmp_path, [a])
    b = write_raw(tmp_path, "b.txt", "two")
    call(tmp_path, [b])
    call(tmp_path, [b])
    tasks = read_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0]["passages"] == ["skyfall-01", "skyfall-02"]


def test_add_files_preserves_unrelated_task_entries(tmp_path):
    tasks_path = tmp_path / "config" / "tasks.yaml"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(
        yaml.safe_dump(
            [{"id": "other-passage", "type": "passage", "passages": ["other-01"]}]
        ),
        encoding="utf-8",
    )
    f = write_raw(tmp_path, "a.txt", "text")
    call(tmp_path, [f])
    tasks = read_tasks(tmp_path)
    assert {t["id"] for t in tasks} == {"other-passage", "skyfall-passage"}


def test_add_files_rejects_unknown_kind(tmp_path):
    f = write_raw(tmp_path, "a.txt", "text")
    with pytest.raises(IntakeError, match="unknown kind"):
        call(tmp_path, [f], kind="poetry")


def test_add_files_rejects_empty_file(tmp_path):
    f = write_raw(tmp_path, "a.txt", "   \n  ")
    with pytest.raises(IntakeError, match="empty"):
        call(tmp_path, [f])
