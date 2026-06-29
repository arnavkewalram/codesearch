"""Embedding engine using bge-small-code-v1 ONNX model."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_REPO = "ArnavKewalram/bge-small-code-v1-onnx"
CACHE_DIR = Path.home() / ".cache" / "codesearch"
MAX_LENGTH = 384
EMBED_DIM = 384


def _download_model() -> Path:
    """Download ONNX model from HuggingFace Hub if not cached."""
    model_dir = CACHE_DIR / "model"
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"

    if model_path.exists() and tokenizer_path.exists():
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL_REPO, "model.onnx", local_dir=str(model_dir))
        hf_hub_download(MODEL_REPO, "tokenizer.json", local_dir=str(model_dir))
    except ImportError:
        # Fallback: try model_quantized.onnx (smaller)
        raise RuntimeError(
            "Install huggingface_hub to auto-download the model:\n"
            "  pip install huggingface_hub\n"
            f"Or manually download from https://huggingface.co/{MODEL_REPO}"
        )

    return model_dir


class CodeEmbedder:
    """Embeds code and queries using bge-small-code-v1 ONNX."""

    def __init__(self, model_dir: str | Path | None = None, quantized: bool = False):
        if model_dir is None:
            model_dir = _download_model()
        model_dir = Path(model_dir)

        model_file = "model_quantized.onnx" if quantized else "model.onnx"
        model_path = model_dir / model_file
        if not model_path.exists():
            model_path = model_dir / "model.onnx"

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding(length=MAX_LENGTH)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts. Returns (N, EMBED_DIM) float32 array."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            # Mean pooling over token embeddings, masked by attention
            token_embeddings = outputs[0]  # (batch, seq_len, dim)
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed = (token_embeddings * mask_expanded).sum(axis=1)
            counts = mask_expanded.sum(axis=1).clip(min=1e-9)
            embeddings = summed / counts

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (EMBED_DIM,) float32 array."""
        return self.embed([text])[0]
