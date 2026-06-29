"""Index a codebase into SQLite with vector embeddings."""
import sqlite3
import time
from pathlib import Path

import numpy as np

from .embedder import CodeEmbedder, EMBED_DIM

INDEX_FILE = ".codesearch.db"

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".lua", ".dart", ".ex", ".exs",
    ".hs", ".ml", ".jl", ".m", ".v", ".sv", ".vhd", ".tf", ".yaml", ".yml",
    ".toml", ".json", ".xml", ".html", ".css", ".scss", ".md", ".rst", ".txt",
}

# Directories to skip
SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    ".env", "dist", "build", ".next", ".nuxt", "target", ".idea", ".vscode",
    ".codesearch", "vendor", "Pods", ".gradle", ".cache",
}

MAX_FILE_SIZE = 500_000  # 500KB
CHUNK_LINES = 30  # lines per chunk
CHUNK_OVERLAP = 5  # overlap between chunks


def _should_index(path: Path) -> bool:
    return path.suffix.lower() in CODE_EXTENSIONS and path.stat().st_size < MAX_FILE_SIZE


def _chunk_file(path: Path, root: Path) -> list[dict]:
    """Split a file into overlapping chunks with metadata."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.splitlines()
    if not lines:
        return []

    rel_path = str(path.relative_to(root))
    chunks = []

    if len(lines) <= CHUNK_LINES:
        chunks.append({
            "path": rel_path,
            "start_line": 1,
            "end_line": len(lines),
            "content": text.strip(),
        })
    else:
        for start in range(0, len(lines), CHUNK_LINES - CHUNK_OVERLAP):
            end = min(start + CHUNK_LINES, len(lines))
            chunk_text = "\n".join(lines[start:end])
            if chunk_text.strip():
                chunks.append({
                    "path": rel_path,
                    "start_line": start + 1,
                    "end_line": end,
                    "content": chunk_text,
                })
            if end >= len(lines):
                break

    return chunks


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    conn.execute("DELETE FROM chunks")  # re-index from scratch
    conn.commit()
    return conn


def index_directory(
    root: str | Path,
    model_dir: str | Path | None = None,
    quantized: bool = False,
    verbose: bool = True,
) -> int:
    """Index a directory. Returns number of chunks indexed."""
    root = Path(root).resolve()
    db_path = root / INDEX_FILE

    if verbose:
        print(f"Indexing {root}...")

    # Collect files
    files = []
    for path in root.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.is_file() and _should_index(path):
            files.append(path)

    if verbose:
        print(f"Found {len(files)} files to index")

    # Chunk files
    all_chunks = []
    for f in files:
        all_chunks.extend(_chunk_file(f, root))

    if not all_chunks:
        print("No code chunks found.")
        return 0

    if verbose:
        print(f"Created {len(all_chunks)} chunks, embedding...")

    # Embed
    t0 = time.time()
    embedder = CodeEmbedder(model_dir=model_dir, quantized=quantized)
    texts = [c["content"] for c in all_chunks]
    embeddings = embedder.embed(texts)
    embed_time = time.time() - t0

    if verbose:
        print(f"Embedded in {embed_time:.1f}s ({len(texts) / embed_time:.0f} chunks/s)")

    # Store
    conn = _init_db(db_path)
    for chunk, emb in zip(all_chunks, embeddings):
        conn.execute(
            "INSERT INTO chunks (path, start_line, end_line, content, embedding) VALUES (?, ?, ?, ?, ?)",
            (chunk["path"], chunk["start_line"], chunk["end_line"], chunk["content"], emb.tobytes()),
        )
    conn.commit()
    conn.close()

    if verbose:
        print(f"Indexed {len(all_chunks)} chunks -> {db_path}")

    return len(all_chunks)
