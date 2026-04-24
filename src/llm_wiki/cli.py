import typer
from typing import Optional
from llm_wiki.db import get_db_connection, ensure_schema_and_model_match
from llm_wiki.sync import run_sync
from llm_wiki.search import search_wiki
import os

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

if __name__ == "__main__":
    app()
