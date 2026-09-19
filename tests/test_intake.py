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


SCRAPED = (
    "괴담에 떨어져도 출근을 해야 하는구나 - 284화\n"
    "\n글자\n-\n16px\n+\n기본\n"
    "괴담에 떨어져도 출근을 해야 하는구나 284화\n"
    "\n괴담에 떨어져도 출근을 해야 하는구나 284화\n"
    "\n“이만큼 탐사 횟수가 쌓였으니.”\n\n호유원의 목소리는 이상했다.\n"
)


def test_title_stem_drops_the_chapter_marker():
    from ati.intake import title_stem

    assert title_stem("괴담에 떨어져도 출근을 해야 하는구나 - 284화") == "괴담에 떨어져도 출근을 해야 하는구나"
    assert title_stem("Some Novel chapter 12") == "Some Novel"


def test_strip_boilerplate_removes_reader_chrome_and_repeated_titles():
    from ati.intake import strip_boilerplate

    out = strip_boilerplate(SCRAPED)
    assert out.startswith("“이만큼")
    assert "16px" not in out
    assert "글자" not in out
    assert "284화" not in out
    assert "호유원의 목소리는 이상했다." in out


def test_strip_boilerplate_leaves_clean_prose_alone():
    from ati.intake import strip_boilerplate

    clean = "그는 검을 뽑았다.\n\n바람이 불었다.\n"
    assert strip_boilerplate(clean) == clean


def test_add_files_strips_headers_before_hashing(tmp_path):
    from ati.corpus import passage_path

    f = write_raw(tmp_path, "chapter_284.txt", SCRAPED)
    call(tmp_path, [f], kind="consistency")
    stored = passage_path(tmp_path / "corpus", "skyfall-ch01").read_text(encoding="utf-8")
    assert "16px" not in stored
    assert stored.startswith("“이만큼")


def test_add_files_can_keep_headers_when_asked(tmp_path):
    from ati.corpus import passage_path
    from ati.intake import add_files

    f = write_raw(tmp_path, "chapter_284.txt", SCRAPED)
    add_files(
        files=[f],
        work="keepme",
        lang="ko",
        genre="x",
        kind="consistency",
        url="u",
        corpus_root=tmp_path / "corpus",
        manifest_path=tmp_path / "corpus" / "manifest.json",
        tasks_path=tmp_path / "config" / "tasks.yaml",
        strip_headers=False,
    )
    stored = passage_path(tmp_path / "corpus", "keepme-ch01").read_text(encoding="utf-8")
    assert "16px" in stored


def test_chars_zero_takes_the_whole_file(tmp_path):
    f = write_raw(tmp_path, "a.txt", "그는 검을 뽑았다. " * 300)
    call(tmp_path, [f], kind="passage", chars=0)
    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["end"] > 800


def test_re_adding_the_source_restores_missing_passage_text(tmp_path):
    """A fresh clone has the manifest but no passage text, and re-adding the
    same files used to be a no-op for anything already registered."""
    f = write_raw(tmp_path, "ch1.txt", "云澈抬起头，看着天空。" * 20)
    first = call(tmp_path, [f], kind="consistency")
    pid = first.added[0]
    target = passage_path(tmp_path / "corpus", pid)
    original = target.read_text(encoding="utf-8")
    manifest_before = read_manifest(tmp_path)

    target.unlink()
    result = call(tmp_path, [f], kind="consistency")
    assert result.restored == [pid] and result.added == [] and result.unchanged == []
    assert target.read_text(encoding="utf-8") == original
    assert read_manifest(tmp_path) == manifest_before


def test_re_adding_repairs_damaged_passage_text(tmp_path):
    f = write_raw(tmp_path, "ch1.txt", "云澈抬起头，看着天空。" * 20)
    pid = call(tmp_path, [f], kind="consistency").added[0]
    target = passage_path(tmp_path / "corpus", pid)
    target.write_text("tampered", encoding="utf-8")
    assert call(tmp_path, [f], kind="consistency").restored == [pid]
    assert sha256_text(target.read_text(encoding="utf-8")) == (
        read_manifest(tmp_path)["passages"][0]["sha256"]
    )


def test_an_intact_passage_is_reported_unchanged_not_restored(tmp_path):
    f = write_raw(tmp_path, "ch1.txt", "云澈抬起头，看着天空。" * 20)
    call(tmp_path, [f], kind="consistency")
    result = call(tmp_path, [f], kind="consistency")
    assert result.unchanged and not result.restored
