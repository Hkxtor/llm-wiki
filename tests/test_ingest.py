import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_wiki.ingest import (
    WikiPageDraft,
    _parse_outline,
    _slugify,
    ingest_source,
    write_drafts,
)


# --- _parse_outline ---

def test_parse_outline_valid():
    data = {"pages": [
        {"category": "concepts", "slug": "llm-wiki", "title": "LLM Wiki", "one_line": "A pattern."},
        {"category": "entities", "slug": "openai", "title": "OpenAI", "one_line": "A company."},
    ]}
    result = _parse_outline(json.dumps(data))
    assert len(result) == 2
    assert result[0]["slug"] == "llm-wiki"


def test_parse_outline_filters_invalid_category():
    data = {"pages": [
        {"category": "invalid", "slug": "foo", "title": "Foo", "one_line": "bar"},
        {"category": "concepts", "slug": "valid", "title": "Valid", "one_line": "ok"},
    ]}
    result = _parse_outline(json.dumps(data))
    assert len(result) == 1
    assert result[0]["slug"] == "valid"


def test_parse_outline_raises_on_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_outline("not json {{{")


def test_parse_outline_empty_pages():
    result = _parse_outline(json.dumps({"pages": []}))
    assert result == []


# --- _slugify ---

def test_slugify_basic():
    assert _slugify("LLM Wiki") == "llm-wiki"


def test_slugify_special_chars():
    assert _slugify("RAG & Vector Search!") == "rag-vector-search"


# --- ingest_source ---

def _make_fake_llm(outline_json: str, page_body: str):
    llm = MagicMock()
    llm.complete.side_effect = [outline_json, page_body]
    return llm


def test_ingest_source_returns_drafts(tmp_wiki_paths, tmp_path):
    source = tmp_path / "test.md"
    source.write_text("# Test\n\nSome meaningful content about LLMs.")

    outline = json.dumps({"pages": [
        {"category": "concepts", "slug": "llm", "title": "LLM", "one_line": "Large language models."},
    ]})
    llm = _make_fake_llm(
        outline,
        "# LLM\n\nLarge language models are neural networks.\n\n## Sources\n- test.md",
    )

    drafts = ingest_source(source, tmp_wiki_paths, llm)
    assert len(drafts) == 1
    assert drafts[0].slug == "llm"
    assert drafts[0].category == "concepts"
    assert drafts[0].title == "LLM"
    assert "## Sources" in drafts[0].body_md


def test_ingest_source_empty_file_returns_empty(tmp_wiki_paths, tmp_path):
    source = tmp_path / "empty.md"
    source.write_text("   ")
    llm = MagicMock()
    result = ingest_source(source, tmp_wiki_paths, llm)
    assert result == []
    llm.complete.assert_not_called()


# --- write_drafts ---

def _make_draft(slug="test-page", category="concepts", title="Test Page"):
    return WikiPageDraft(
        category=category,
        slug=slug,
        title=title,
        body_md="# Test Page\n\nContent here.\n\n## Sources\n- raw/test.md",
        source_refs=("raw/test.md",),
    )


def test_write_drafts_creates_new_file(tmp_wiki_paths):
    draft = _make_draft()
    written = write_drafts([draft], tmp_wiki_paths)
    target = tmp_wiki_paths.concepts / "test-page.md"
    assert target.exists()
    assert "# Test Page" in target.read_text()
    assert written == [target]


def test_write_drafts_merges_existing(tmp_wiki_paths):
    draft = _make_draft()
    target = tmp_wiki_paths.concepts / "test-page.md"
    target.write_text("# Test Page\n\nOriginal content.\n")

    write_drafts([draft], tmp_wiki_paths, overwrite=False)
    content = target.read_text()
    assert "Original content" in content
    assert "## Update" in content


def test_write_drafts_overwrites_with_backup(tmp_wiki_paths):
    draft = _make_draft()
    target = tmp_wiki_paths.concepts / "test-page.md"
    target.write_text("# Test Page\n\nOld content.\n")

    write_drafts([draft], tmp_wiki_paths, overwrite=True)
    assert "Old content" not in target.read_text()
    trash_files = list((tmp_wiki_paths.wiki / ".wiki-trash").glob("test-page-*.md"))
    assert len(trash_files) == 1


def test_write_drafts_dry_run_does_not_write(tmp_wiki_paths, capsys):
    draft = _make_draft()
    write_drafts([draft], tmp_wiki_paths, dry_run=True)
    target = tmp_wiki_paths.concepts / "test-page.md"
    assert not target.exists()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


def test_write_drafts_entity_goes_to_entities_dir(tmp_wiki_paths):
    draft = _make_draft(slug="openai", category="entities", title="OpenAI")
    write_drafts([draft], tmp_wiki_paths)
    assert (tmp_wiki_paths.entities / "openai.md").exists()
