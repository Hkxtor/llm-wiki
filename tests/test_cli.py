from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from llm_wiki.cli import app
from llm_wiki.ingest import WikiPageDraft
from llm_wiki.maintenance import LintReport

runner = CliRunner()


@patch("llm_wiki.cli.get_db_connection")
@patch("llm_wiki.cli.ensure_schema_and_model_match")
def test_init_command(mock_ensure, mock_conn):
    mock_conn.return_value = MagicMock()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    mock_ensure.assert_called_once()


@patch("llm_wiki.cli.run_sync")
def test_sync_command(mock_run_sync):
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    mock_run_sync.assert_called_once()


@patch("llm_wiki.cli.search_wiki")
def test_search_command_no_results(mock_search):
    mock_search.return_value = []
    result = runner.invoke(app, ["search", "test query"])
    assert result.exit_code == 0
    assert "No results found" in result.output


@patch("llm_wiki.cli.search_wiki")
def test_search_command_with_results(mock_search):
    mock_search.return_value = [
        {"file_path": "wiki/concepts/foo.md", "content": "foo content", "distance": 0.1}
    ]
    result = runner.invoke(app, ["search", "test query"])
    assert result.exit_code == 0
    assert "foo.md" in result.output


@patch("llm_wiki.cli.search_wiki")
def test_search_command_with_filter(mock_search):
    mock_search.return_value = []
    result = runner.invoke(app, ["search", "query", "--filter", "wiki/concepts"])
    assert result.exit_code == 0
    mock_search.assert_called_once_with("query", top_k=5, path_filter="wiki/concepts")


@patch("llm_wiki.cli.run_sync")
@patch("llm_wiki.cli.append_log")
@patch("llm_wiki.cli.write_drafts")
@patch("llm_wiki.cli.ingest_source")
@patch("llm_wiki.cli.OpenAICompatibleLLM")
@patch("llm_wiki.cli.WikiPaths")
@patch("llm_wiki.cli.Config")
def test_ingest_command_writes_pages(
    mock_cfg, mock_paths_cls, mock_llm_cls, mock_ingest, mock_write, mock_log, mock_sync, tmp_path
):
    source = tmp_path / "test.md"
    source.write_text("# Test content")

    mock_cfg.from_env.return_value = MagicMock()
    paths = MagicMock()
    paths.wiki = tmp_path / "wiki"
    paths.concepts = tmp_path / "wiki" / "concepts"
    mock_paths_cls.from_config.return_value = paths

    draft = WikiPageDraft(
        category="concepts", slug="test", title="Test",
        body_md="# Test\n\nBody.", source_refs=(str(source),)
    )
    mock_ingest.return_value = [draft]

    target = tmp_path / "wiki" / "concepts" / "test.md"
    mock_write.return_value = [target]

    result = runner.invoke(app, ["ingest", str(source), "--yes"])
    assert result.exit_code == 0
    mock_write.assert_called_once()
    mock_log.assert_called_once()


@patch("llm_wiki.cli.update_index")
@patch("llm_wiki.cli.WikiPaths")
@patch("llm_wiki.cli.Config")
def test_reindex_command(mock_cfg, mock_paths_cls, mock_update_index):
    mock_cfg.from_env.return_value = MagicMock()
    paths = MagicMock()
    mock_paths_cls.from_config.return_value = paths

    result = runner.invoke(app, ["reindex"])
    assert result.exit_code == 0
    mock_update_index.assert_called_once_with(paths)


@patch("llm_wiki.cli.lint_wiki")
@patch("llm_wiki.cli.WikiPaths")
@patch("llm_wiki.cli.Config")
def test_lint_command_no_issues(mock_cfg, mock_paths_cls, mock_lint):
    mock_cfg.from_env.return_value = MagicMock()
    mock_paths_cls.from_config.return_value = MagicMock()
    mock_lint.return_value = LintReport(orphans=(), broken_links=(), missing_titles=(), empty_pages=())

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0
    assert "No issues" in result.output


@patch("llm_wiki.cli.append_log")
@patch("llm_wiki.cli.lint_wiki")
@patch("llm_wiki.cli.WikiPaths")
@patch("llm_wiki.cli.Config")
def test_lint_command_with_issues(mock_cfg, mock_paths_cls, mock_lint, mock_log):
    mock_cfg.from_env.return_value = MagicMock()
    mock_paths_cls.from_config.return_value = MagicMock()
    mock_lint.return_value = LintReport(
        orphans=("orphan.md",), broken_links=(), missing_titles=(), empty_pages=()
    )

    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 1
    assert "orphan.md" in result.output
    mock_log.assert_called_once()


@patch("llm_wiki.cli.Config")
def test_ingest_command_file_not_found(mock_cfg, tmp_path):
    mock_cfg.from_env.return_value = MagicMock()
    result = runner.invoke(app, ["ingest", str(tmp_path / "missing.md")])
    assert result.exit_code == 1


@patch("llm_wiki.cli.run_sync")
@patch("llm_wiki.cli.append_log")
@patch("llm_wiki.cli.write_drafts")
@patch("llm_wiki.cli.ingest_source")
@patch("llm_wiki.cli.OpenAICompatibleLLM")
@patch("llm_wiki.cli.WikiPaths")
@patch("llm_wiki.cli.Config")
def test_ingest_command_dry_run(
    mock_cfg, mock_paths_cls, mock_llm_cls, mock_ingest, mock_write, mock_log, mock_sync, tmp_path
):
    source = tmp_path / "test.md"
    source.write_text("content")

    mock_cfg.from_env.return_value = MagicMock()
    paths = MagicMock()
    paths.wiki = tmp_path / "wiki"
    paths.concepts = tmp_path / "wiki" / "concepts"
    mock_paths_cls.from_config.return_value = paths

    draft = WikiPageDraft(
        category="concepts", slug="test", title="Test",
        body_md="# Test", source_refs=(str(source),)
    )
    mock_ingest.return_value = [draft]
    mock_write.return_value = [tmp_path / "wiki" / "concepts" / "test.md"]

    result = runner.invoke(app, ["ingest", str(source), "--dry-run"])
    assert result.exit_code == 0
    mock_log.assert_not_called()
    mock_sync.assert_not_called()
