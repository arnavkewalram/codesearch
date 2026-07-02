# semantic-codesearch

Semantic code search CLI — find code by meaning, not just text.

Powered by [bge-small-code-v1](https://huggingface.co/ArnavKewalram/bge-small-code-v1), a 33M parameter code embedding model trained on 200K CoRNStack triplets across Python, JavaScript, Java, and Go.

## Install

```bash
pip install semantic-codesearch
```

## Usage

### Index a codebase

```bash
codesearch index .
```

Walks the directory, chunks code files (30 lines with 5-line overlap), embeds each chunk with the ONNX model, and stores everything in a local SQLite database (`.codesearch.db`).

### Search by meaning

```bash
codesearch search "function that sorts users by date"
codesearch search "authentication middleware" -n 10
codesearch search "database connection pool" -d /path/to/repo
```

Results show file path, line range, similarity score, and a code preview:

```
────────────────────────────────────────────────────────────
  #1 src/auth.py:26-32  (71.8% match)
────────────────────────────────────────────────────────────
    26 if AUTH_TOKEN:
    27     auth = request.headers.get("authorization", "")
    28     if auth != f"Bearer {AUTH_TOKEN}":
    29         return JSONResponse({"error": "unauthorized"}, ...)
```

### View index stats

```bash
codesearch stats
```

## Features

- **Semantic search** — finds code by meaning, not keywords. "sort by date" finds `sorted(users, key=lambda u: u.created_at)`.
- **Fast** — ONNX model runs on CPU. Indexing ~50 files takes ~15 seconds. Searches are instant (cosine similarity on cached embeddings).
- **Local & private** — everything runs locally. No API calls, no data leaves your machine.
- **Auto-downloads model** — fetches bge-small-code-v1 ONNX from HuggingFace on first run (~130MB).
- **50+ file types** — Python, JS, TS, Java, Go, Rust, C/C++, SQL, YAML, and more.
- **Smart directory skipping** — ignores `.git`, `node_modules`, `__pycache__`, `.venv`, `dist`, etc.

## How it works

1. **Chunking** — splits each file into overlapping 30-line chunks
2. **Embedding** — runs each chunk through bge-small-code-v1 (ONNX, 384-dim output)
3. **Storage** — stores embeddings + metadata in SQLite (`.codesearch.db`)
4. **Search** — embeds your query, computes cosine similarity against all chunks, returns top-k

## Model

Built on [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (33M params), fine-tuned on [CoRNStack](https://huggingface.co/collections/nomic-ai/cornstack) code search triplets with Matryoshka loss for flexible embedding dimensions (384/256/128/64).

- **Accuracy@1**: 72.6% | **Accuracy@10**: 91.8% | **NDCG@10**: 82.5%
- **ONNX INT8**: 33.8MB — small enough to run in a browser

## Requirements

- Python 3.10-3.13 (onnxruntime doesn't support 3.14 yet)
- No GPU needed — runs on CPU

## License

MIT
