from unittest.mock import MagicMock, patch

from llm_wiki.search import search_wiki


def _make_cursor_mock(rows):
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall = MagicMock(return_value=rows)
    return cursor


@patch("llm_wiki.search.get_db_connection")
@patch("llm_wiki.search.OpenAIEmbedder")
def test_search_returns_results(mock_embedder_cls, mock_get_conn):
    mock_embedder_cls.return_value.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    rows = [{"file_path": "wiki/concepts/foo.md", "content": "foo text", "distance": 0.1}]
    cursor = _make_cursor_mock(rows)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_get_conn.return_value = conn

    results = search_wiki("test query", top_k=1)

    assert len(results) == 1
    assert results[0]["file_path"] == "wiki/concepts/foo.md"


@patch("llm_wiki.search.get_db_connection")
@patch("llm_wiki.search.OpenAIEmbedder")
def test_search_with_path_filter_adds_where_clause(mock_embedder_cls, mock_get_conn):
    mock_embedder_cls.return_value.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    cursor = _make_cursor_mock([])
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_get_conn.return_value = conn

    search_wiki("query", top_k=5, path_filter="wiki/concepts")

    executed_sql = cursor.execute.call_args[0][0]
    assert "LIKE" in executed_sql


@patch("llm_wiki.search.get_db_connection")
@patch("llm_wiki.search.OpenAIEmbedder")
def test_search_returns_empty_on_db_error(mock_embedder_cls, mock_get_conn):
    mock_embedder_cls.return_value.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    conn = MagicMock()
    conn.cursor.side_effect = Exception("db error")
    mock_get_conn.return_value = conn

    results = search_wiki("query")
    assert results == []


@patch("llm_wiki.search.get_db_connection")
@patch("llm_wiki.search.OpenAIEmbedder")
def test_search_uses_parameterized_query(mock_embedder_cls, mock_get_conn):
    mock_embedder_cls.return_value.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    cursor = _make_cursor_mock([])
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_get_conn.return_value = conn

    search_wiki("query", top_k=3)

    executed_sql = cursor.execute.call_args[0][0]
    params = cursor.execute.call_args[0][1]
    # Must use %s placeholders, not f-string interpolation
    assert "%s" in executed_sql
    assert isinstance(params, list)
    assert len(params) >= 1


@patch("llm_wiki.search.get_db_connection")
@patch("llm_wiki.search.OpenAIEmbedder")
def test_search_returns_empty_on_embedding_error(mock_embedder_cls, mock_get_conn):
    mock_embedder_cls.return_value.embed.side_effect = Exception("api error")
    results = search_wiki("query")
    assert results == []
