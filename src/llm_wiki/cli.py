from pathlib import Path
from typing import Optional

import typer

from llm_wiki.config import Config
from llm_wiki.db import ensure_schema_and_model_match, get_db_connection
from llm_wiki.ingest import ingest_source, write_drafts
from llm_wiki.llm import OpenAICompatibleLLM
from llm_wiki.maintenance import append_log, lint_wiki, update_index
from llm_wiki.paths import WikiPaths
from llm_wiki.search import search_wiki
from llm_wiki.sync import run_sync

app = typer.Typer(help="LLM Wiki CLI - Sync and Search your knowledge base.")

@app.command()
def init():
    """Initialize the OceanBase database schema."""
    from llm_wiki.db import EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP
    conn = get_db_connection()
    ensure_schema_and_model_match(conn, EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP)
    conn.close()

@app.command()
def sync():
    """Sync local markdown files to OceanBase."""
    run_sync()

@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="Number of results to return"),
    path_filter: Optional[str] = typer.Option(None, "--filter", help="Filter by file path (e.g. 'wiki/concepts')")
):
    """Search the wiki for the given query."""
    results = search_wiki(query, top_k=k, path_filter=path_filter)
    
    if not results:
        typer.echo("No results found.")
        return

    typer.echo("\n" + "="*80)
    typer.echo(f"TOP {len(results)} RESULTS FOR: '{query}'")
    typer.echo("="*80 + "\n")

    for i, r in enumerate(results, 1):
        typer.echo(f"[{i}] File: {r['file_path']} (Distance: {r['distance']:.4f})")
        typer.echo("-" * 40)
        typer.echo(r['content'])
        typer.echo("=" * 80 + "\n")

@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Path to source file in raw/ directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing pages instead of merging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    auto_sync: bool = typer.Option(True, "--auto-sync/--no-auto-sync", help="Run sync after writing"),
):
    """Ingest a source file from raw/ and generate wiki pages via LLM."""
    cfg = Config.from_env()
    paths = WikiPaths.from_config(cfg)
    paths.ensure_exists()

    if not source.exists():
        typer.echo(f"[X] File not found: {source}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracting outline from {source.name}...")
    llm = OpenAICompatibleLLM(cfg)
    drafts = ingest_source(source, paths, llm)

    if not drafts:
        typer.echo("No pages extracted from source.")
        raise typer.Exit(0)

    typer.echo(f"\nPages to {'preview' if dry_run else 'write'} ({len(drafts)}):")
    for d in drafts:
        target = paths.wiki / d.category / f"{d.slug}.md"
        status = "new" if not target.exists() else ("overwrite" if overwrite else "merge")
        typer.echo(f"  [{status}] wiki/{d.category}/{d.slug}.md — {d.title}")

    if not dry_run and not yes:
        typer.confirm("\nProceed?", abort=True)

    written = write_drafts(drafts, paths, overwrite=overwrite, dry_run=dry_run)

    if not dry_run:
        append_log(paths, "ingest", source.name)
        typer.echo(f"\n[*] Wrote {len(written)} pages. Log updated.")
        if auto_sync:
            typer.echo("Running sync...")
            run_sync(paths=paths)
    else:
        typer.echo(f"\n[dry-run] Would write {len(written)} pages.")


@app.command()
def reindex():
    """Regenerate wiki/index.md from all wiki pages."""
    cfg = Config.from_env()
    paths = WikiPaths.from_config(cfg)
    paths.ensure_exists()
    update_index(paths)
    typer.echo("[*] wiki/index.md updated.")


@app.command()
def lint():
    """Check wiki for orphans, broken links, missing titles, and empty pages."""
    cfg = Config.from_env()
    paths = WikiPaths.from_config(cfg)

    report = lint_wiki(paths)

    if not report.has_issues:
        typer.echo("[OK] No issues found.")
        return

    if report.orphans:
        typer.echo(f"\nOrphan pages ({len(report.orphans)}):")
        for name in report.orphans:
            typer.echo(f"  - {name}")

    if report.broken_links:
        typer.echo(f"\nBroken links ({len(report.broken_links)}):")
        for entry in report.broken_links:
            typer.echo(f"  - {entry}")

    if report.missing_titles:
        typer.echo(f"\nMissing H1 title ({len(report.missing_titles)}):")
        for name in report.missing_titles:
            typer.echo(f"  - {name}")

    if report.empty_pages:
        typer.echo(f"\nEmpty pages ({len(report.empty_pages)}):")
        for name in report.empty_pages:
            typer.echo(f"  - {name}")

    append_log(paths, "lint", "wiki check", details=str(report))
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
