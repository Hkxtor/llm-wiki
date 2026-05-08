from llm_wiki.maintenance import append_log


def test_append_log_creates_log_file(tmp_wiki_paths):
    assert not tmp_wiki_paths.log_md.exists()
    append_log(tmp_wiki_paths, "ingest", "llm-wiki.md")
    assert tmp_wiki_paths.log_md.exists()


def test_append_log_format(tmp_wiki_paths):
    append_log(tmp_wiki_paths, "ingest", "llm-wiki.md")
    content = tmp_wiki_paths.log_md.read_text()
    assert "## [" in content
    assert "] ingest | llm-wiki.md" in content


def test_append_log_appends_multiple_entries(tmp_wiki_paths):
    append_log(tmp_wiki_paths, "ingest", "file1.md")
    append_log(tmp_wiki_paths, "query", "what is RAG?")
    content = tmp_wiki_paths.log_md.read_text()
    assert "ingest | file1.md" in content
    assert "query | what is RAG?" in content


def test_append_log_with_details(tmp_wiki_paths):
    append_log(tmp_wiki_paths, "lint", "wiki check", details="Found 2 orphan pages")
    content = tmp_wiki_paths.log_md.read_text()
    assert "Found 2 orphan pages" in content


def test_append_log_preserves_existing_content(tmp_wiki_paths):
    tmp_wiki_paths.log_md.write_text("## [2026-01-01 00:00] ingest | old.md\n\n")
    append_log(tmp_wiki_paths, "ingest", "new.md")
    content = tmp_wiki_paths.log_md.read_text()
    assert "old.md" in content
    assert "new.md" in content
