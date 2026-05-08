from llm_wiki.maintenance import LintReport, lint_wiki, update_index


def test_update_index_creates_index_md(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "llm.md").write_text("# LLM\n\nContent.\n")
    (tmp_wiki_paths.entities / "openai.md").write_text("# OpenAI\n\nContent.\n")

    update_index(tmp_wiki_paths)

    assert tmp_wiki_paths.index_md.exists()
    content = tmp_wiki_paths.index_md.read_text()
    assert "[[LLM]]" in content
    assert "[[OpenAI]]" in content


def test_update_index_groups_by_category(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "rag.md").write_text("# RAG\n\nContent.\n")
    (tmp_wiki_paths.synthesis / "overview.md").write_text("# Overview\n\nContent.\n")

    update_index(tmp_wiki_paths)
    content = tmp_wiki_paths.index_md.read_text()

    assert "## Concepts" in content
    assert "## Synthesis" in content
    assert "## Entities" not in content


def test_update_index_empty_wiki(tmp_wiki_paths):
    update_index(tmp_wiki_paths)
    content = tmp_wiki_paths.index_md.read_text()
    assert "# Wiki Index" in content


def test_update_index_uses_file_stem_when_no_title(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "no-title.md").write_text("Some content without H1.\n")

    update_index(tmp_wiki_paths)
    content = tmp_wiki_paths.index_md.read_text()
    assert "[[no-title]]" in content


def test_lint_wiki_no_broken_links_when_all_exist(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "alpha.md").write_text(
        "# Alpha\n\n[[Beta]]\n\nContent here.\n"
    )
    (tmp_wiki_paths.concepts / "beta.md").write_text(
        "# Beta\n\n[[Alpha]]\n\nContent here.\n"
    )

    report = lint_wiki(tmp_wiki_paths)
    assert not report.broken_links
    assert not report.missing_titles
    assert not report.empty_pages


def test_lint_wiki_detects_broken_links(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "lone.md").write_text(
        "# Lone\n\n[[NonExistent]]\n\nContent here.\n"
    )

    report = lint_wiki(tmp_wiki_paths)
    assert any("NonExistent" in b for b in report.broken_links)


def test_lint_wiki_detects_missing_title(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "notitle.md").write_text("No heading here.\n\nContent.\n")

    report = lint_wiki(tmp_wiki_paths)
    assert "notitle.md" in report.missing_titles


def test_lint_wiki_detects_empty_pages(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "empty.md").write_text("# Empty\n")

    report = lint_wiki(tmp_wiki_paths)
    assert "empty.md" in report.empty_pages


def test_lint_report_has_issues_false_when_clean():
    report = LintReport(orphans=(), broken_links=(), missing_titles=(), empty_pages=())
    assert not report.has_issues


def test_lint_report_has_issues_true_when_broken_links():
    report = LintReport(
        orphans=(), broken_links=("foo.md: [[Bar]]",), missing_titles=(), empty_pages=()
    )
    assert report.has_issues


def test_lint_wiki_orphan_detection(tmp_wiki_paths):
    (tmp_wiki_paths.concepts / "alpha.md").write_text("# Alpha\n\nContent.\n")
    (tmp_wiki_paths.concepts / "beta.md").write_text("# Beta\n\n[[Alpha]]\n\nContent.\n")
    (tmp_wiki_paths.concepts / "gamma.md").write_text("# Gamma\n\nContent.\n")

    report = lint_wiki(tmp_wiki_paths)
    assert "gamma.md" in report.orphans
    assert "alpha.md" not in report.orphans
