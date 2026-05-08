from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_wiki.sync import (
    calculate_file_hash,
    chunk_text,
    extract_text_from_file,
    get_existing_documents,
    run_sync,
    sync_file,
)
from tests.conftest import make_config


# --- calculate_file_hash ---

def test_hash_returns_string(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("hello")
    result = calculate_file_hash(str(f))
    assert isinstance(result, str)
    assert len(result) > 0


def test_hash_is_deterministic(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("hello world")
    assert calculate_file_hash(str(f)) == calculate_file_hash(str(f))


def test_hash_differs_for_different_content(tmp_path: Path):
    f1, f2 = tmp_path / "a.md", tmp_path / "b.md"
    f1.write_text("hello")
    f2.write_text("world")
    assert calculate_file_hash(str(f1)) != calculate_file_hash(str(f2))


# --- chunk_text ---

def test_chunk_text_returns_list():
    chunks = chunk_text("This is a sentence. And another one.", chunk_size=20, overlap=2)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_text_empty_input():
    result = chunk_text("")
    assert isinstance(result, list)


# --- extract_text_from_file ---

def test_extract_markdown(tmp_path: Path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nSome content here.")
    result = extract_text_from_file(str(f))
    assert "Title" in result
    assert "Some content" in result


def test_extract_unsupported_extension(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("ignored")
    assert extract_text_from_file(str(f)) == ""


# --- get_existing_documents ---

def test_get_existing_documents_returns_dict(mock_db_conn):
    mock_db_conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        {"file_path": "wiki/concepts/foo.md", "file_hash": "abc123"},
    ]
    result = get_existing_documents(mock_db_conn)
    assert result == {"wiki/concepts/foo.md": "abc123"}


def test_get_existing_documents_empty(mock_db_conn):
    mock_db_conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
    assert get_existing_documents(mock_db_conn) == {}


# --- sync_file ---

def test_sync_file_skips_empty_content(tmp_path: Path, mock_db_conn, fake_embedder):
    f = tmp_path / "empty.md"
    f.write_text("   ")
    sync_file(mock_db_conn, str(f), embedder=fake_embedder)
    mock_db_conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()


def test_sync_file_inserts_doc_and_chunks(tmp_path: Path, mock_db_conn, fake_embedder):
    f = tmp_path / "llm.md"
    f.write_text("# LLM\n\nLarge language models are neural networks trained on text data.")
    sync_file(mock_db_conn, str(f), embedder=fake_embedder)
    cursor = mock_db_conn.cursor.return_value.__enter__.return_value
    calls = [str(c) for c in cursor.execute.call_args_list]
    assert any("DELETE" in c for c in calls)
    assert any("wiki_documents" in c for c in calls)


# --- run_sync ---

@patch("llm_wiki.sync.ensure_schema_and_model_match")
@patch("llm_wiki.sync.get_db_connection")
@patch("llm_wiki.sync.Config")
def test_run_sync_processes_new_file(mock_cfg_cls, mock_get_conn, mock_ensure, tmp_wiki_paths, fake_embedder):
    mock_cfg_cls.from_env.return_value = make_config()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
    mock_get_conn.return_value = conn

    # Create a wiki file
    md = tmp_wiki_paths.concepts / "test.md"
    md.write_text("# Test\n\nSome content for testing sync.")

    run_sync(paths=tmp_wiki_paths, embedder=fake_embedder)

    cursor = conn.cursor.return_value.__enter__.return_value
    all_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "wiki_documents" in all_calls


@patch("llm_wiki.sync.ensure_schema_and_model_match")
@patch("llm_wiki.sync.get_db_connection")
@patch("llm_wiki.sync.Config")
def test_run_sync_skips_unchanged_file(mock_cfg_cls, mock_get_conn, mock_ensure, tmp_wiki_paths, fake_embedder):
    mock_cfg_cls.from_env.return_value = make_config()
    conn = MagicMock()

    md = tmp_wiki_paths.concepts / "unchanged.md"
    md.write_text("# Stable\n\nContent that hasn't changed.")
    existing_hash = calculate_file_hash(str(md))

    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        {"file_path": str(md).replace("\\", "/"), "file_hash": existing_hash}
    ]
    mock_get_conn.return_value = conn

    run_sync(paths=tmp_wiki_paths, embedder=fake_embedder)

    cursor = conn.cursor.return_value.__enter__.return_value
    all_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "INSERT" not in all_calls
