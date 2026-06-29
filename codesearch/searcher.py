"""Search indexed code by semantic similarity."""
import sqlite3
from pathlib import Path

import numpy as np

from .embedder import CodeEmbedder, EMBED_DIM
from .indexer import INDEX_FILE


def search(
    query: str,
    root: str | Path = ".",
    top_k: int = 5,
    model_dir: str | Path | None = None,
    quantized: bool = False,
) -> list[dict]:
    """Search for code semantically similar to query.

    Returns list of dicts with: path, start_line, end_line, content, score
    """
    root = Path(root).resolve()
    db_path = root / INDEX_FILE

    if not db_path.exists():
        raise FileNotFoundError(
            f"No index found at {db_path}. Run 'codesearch index' first."
        )

    # Embed query
    embedder = CodeEmbedder(model_dir=model_dir, quantized=quantized)
    query_emb = embedder.embed_one(query)

    # Load all embeddings from DB
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, path, start_line, end_line, content, embedding FROM chunks"
    ).fetchall()
    conn.close()

    if not rows:
        return []

    # Compute cosine similarity (embeddings are already L2-normalized)
    ids, paths, starts, ends, contents, emb_blobs = zip(*rows)
    embeddings = np.array(
        [np.frombuffer(b, dtype=np.float32) for b in emb_blobs]
    )

    scores = embeddings @ query_emb  # cosine similarity (normalized vectors)

    # Top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "path": paths[idx],
            "start_line": starts[idx],
            "end_line": ends[idx],
            "content": contents[idx],
            "score": float(scores[idx]),
        })

    return results
