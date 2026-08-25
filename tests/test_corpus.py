import json

import pytest

from ati.corpus import (
    HashMismatch,
    PassageMissing,
    load_manifest,
    load_passage,
    passage_path,
    sha256_text,
)


def test_sha256_is_stable_and_utf8_based():
    assert sha256_text("霜降") == sha256_text("霜降")
    assert sha256_text("a") != sha256_text("b")
    assert len(sha256_text("a")) == 64


def test_load_manifest_parses_entries(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "passages": [
                    {
                        "id": "cn-01",
                        "lang": "zh",
                        "genre": "xianxia",
                        "source_url": "https://example.invalid/book/1",
                        "chapter": "1",
                        "start": 0,
                        "end": 800,
                        "sha256": "0" * 64,
                        "has_reference": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    passages = load_manifest(manifest)
    assert len(passages) == 1
    assert passages[0].id == "cn-01"
    assert passages[0].lang == "zh"
    assert passages[0].has_reference is False
    assert passages[0].series is None


def test_load_manifest_rejects_duplicate_passage_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    entry = {
        "id": "dupe",
        "lang": "zh",
        "genre": "x",
        "source_url": "u",
        "chapter": "1",
        "start": 0,
        "end": 1,
        "sha256": "0" * 64,
        "has_reference": False,
    }
    manifest.write_text(json.dumps({"passages": [entry, entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate passage id"):
        load_manifest(manifest)


def test_load_passage_returns_text_when_hash_matches(tmp_path):
    from ati.corpus import Passage

    text = "the quick brown fox"
    p = Passage(
        id="ctrl-01",
        lang="en",
        genre="control",
        source_url="u",
        chapter="1",
        start=0,
        end=len(text),
        sha256=sha256_text(text),
        has_reference=True,
    )
    passage_path(tmp_path, p.id).parent.mkdir(parents=True, exist_ok=True)
    passage_path(tmp_path, p.id).write_text(text, encoding="utf-8")
    assert load_passage(tmp_path, p) == text


def test_load_passage_raises_on_hash_mismatch(tmp_path):
    from ati.corpus import Passage

    p = Passage(
        id="cn-01",
        lang="zh",
        genre="x",
        source_url="u",
        chapter="1",
        start=0,
        end=3,
        sha256=sha256_text("original"),
        has_reference=False,
    )
    passage_path(tmp_path, p.id).parent.mkdir(parents=True, exist_ok=True)
    passage_path(tmp_path, p.id).write_text("changed", encoding="utf-8")
    with pytest.raises(HashMismatch, match="cn-01"):
        load_passage(tmp_path, p)


def test_load_passage_raises_when_file_absent(tmp_path):
    from ati.corpus import Passage

    p = Passage(
        id="jp-01",
        lang="ja",
        genre="x",
        source_url="u",
        chapter="1",
        start=0,
        end=3,
        sha256="0" * 64,
        has_reference=False,
    )
    with pytest.raises(PassageMissing, match="jp-01"):
        load_passage(tmp_path, p)
