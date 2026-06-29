"""CLI for codesearch."""
import sys

import click


@click.group()
@click.version_option()
def main():
    """Semantic code search — find code by meaning, not just text."""
    pass


@main.command()
@click.argument("directory", default=".")
@click.option("--quantized", is_flag=True, help="Use INT8 quantized model (smaller, faster)")
def index(directory, quantized):
    """Index a codebase for semantic search."""
    from .indexer import index_directory
    count = index_directory(directory, quantized=quantized)
    if count:
        click.echo(f"\nReady. Run 'codesearch search \"your query\"' to search.")


@main.command()
@click.argument("query")
@click.option("-n", "--top", default=5, help="Number of results (default: 5)")
@click.option("-d", "--directory", default=".", help="Directory to search (default: .)")
@click.option("--quantized", is_flag=True, help="Use INT8 quantized model")
@click.option("--no-preview", is_flag=True, help="Don't show code preview")
def search(query, top, directory, quantized, no_preview):
    """Search for code semantically similar to a query."""
    from .searcher import search as do_search

    try:
        results = do_search(query, root=directory, top_k=top, quantized=quantized)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score_pct = r["score"] * 100
        click.echo(f"\n{'─' * 60}")
        click.echo(
            click.style(f"  #{i} ", fg="cyan", bold=True)
            + click.style(f"{r['path']}", fg="green")
            + f":{r['start_line']}-{r['end_line']}"
            + click.style(f"  ({score_pct:.1f}% match)", fg="yellow")
        )
        click.echo(f"{'─' * 60}")

        if not no_preview:
            lines = r["content"].splitlines()
            preview = lines[:15]
            for ln_num, line in enumerate(preview, r["start_line"]):
                click.echo(click.style(f"  {ln_num:4d} ", fg="bright_black") + line)
            if len(lines) > 15:
                click.echo(click.style(f"       ... ({len(lines) - 15} more lines)", fg="bright_black"))

    click.echo(f"\n{len(results)} results for: \"{query}\"")


@main.command()
@click.argument("directory", default=".")
def stats(directory):
    """Show index statistics."""
    import sqlite3
    from pathlib import Path
    from .indexer import INDEX_FILE

    db_path = Path(directory).resolve() / INDEX_FILE
    if not db_path.exists():
        click.echo("No index found. Run 'codesearch index' first.")
        return

    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    files = conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
    conn.close()

    size_mb = db_path.stat().st_size / 1e6
    click.echo(f"Index: {db_path}")
    click.echo(f"Files: {files}")
    click.echo(f"Chunks: {total}")
    click.echo(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
