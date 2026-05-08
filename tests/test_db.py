import pymysql
import pytest
from unittest.mock import MagicMock, call, patch

from llm_wiki.db import get_config, set_config, initialize_base_schema, ensure_schema_and_model_match


def test_get_config_returns_value(mock_db_conn):
    mock_db_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = {
        "config_value": "text-embedding-3-small"
    }
    result = get_config(mock_db_conn, "ACTIVE_EMBEDDING_MODEL")
    assert result == "text-embedding-3-small"


def test_get_config_returns_none_when_key_missing(mock_db_conn):
    mock_db_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    result = get_config(mock_db_conn, "NONEXISTENT_KEY")
    assert result is None


def test_get_config_returns_none_on_table_not_exist(mock_db_conn):
    mock_db_conn.cursor.return_value.__enter__.side_effect = pymysql.err.ProgrammingError(
        1146, "Table doesn't exist"
    )
    result = get_config(mock_db_conn, "ACTIVE_EMBEDDING_MODEL")
    assert result is None


def test_get_config_raises_on_unexpected_db_error(mock_db_conn):
    mock_db_conn.cursor.return_value.__enter__.side_effect = Exception("connection reset")
    with pytest.raises(RuntimeError, match="Failed to read config key"):
        get_config(mock_db_conn, "ACTIVE_EMBEDDING_MODEL")


def test_set_config_executes_upsert(mock_db_conn):
    set_config(mock_db_conn, "ACTIVE_CHUNK_SIZE", "800")
    cursor = mock_db_conn.cursor.return_value.__enter__.return_value
    sql = cursor.execute.call_args[0][0]
    assert "INSERT" in sql
    assert "ON DUPLICATE KEY UPDATE" in sql


def test_initialize_base_schema_creates_tables(mock_db_conn):
    initialize_base_schema(mock_db_conn)
    cursor = mock_db_conn.cursor.return_value.__enter__.return_value
    all_sql = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "wiki_config" in all_sql
    assert "wiki_documents" in all_sql


def test_ensure_schema_first_time_setup(mock_db_conn):
    # get_config returns None = first time setup
    mock_db_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    ensure_schema_and_model_match(mock_db_conn, "text-embedding-3-small", 1536, 800, 100)
    cursor = mock_db_conn.cursor.return_value.__enter__.return_value
    all_sql = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "wiki_chunks" in all_sql


def test_ensure_schema_up_to_date(mock_db_conn):
    # Simulate all config values matching current config
    responses = [
        {"config_value": "text-embedding-3-small"},
        {"config_value": "800"},
        {"config_value": "100"},
    ]
    mock_db_conn.cursor.return_value.__enter__.return_value.fetchone.side_effect = responses
    # Should not drop or recreate tables — just print up-to-date message
    ensure_schema_and_model_match(mock_db_conn, "text-embedding-3-small", 1536, 800, 100)
    cursor = mock_db_conn.cursor.return_value.__enter__.return_value
    all_sql = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "DROP" not in all_sql
