"""Local embedding helpers."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import numpy as np


DEFAULT_MODEL = "BAAI/bge-m3"
FALLBACK_MODEL = "local-hashing-1024"
TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def configured_embedding_model() -> str:
    from .cli import load_config

    return str(load_config().get("embedding_model", DEFAULT_MODEL))


class Embedder:
    """Encode text with bge-m3 when available, otherwise a local test fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int = 16,
        device: str | None = None,
        allow_hash_fallback: bool = True,
    ) -> None:
        resolved_model = model_name or configured_embedding_model()
        self.requested_model_name = resolved_model
        self.batch_size = batch_size
        self.device = device or detect_device()
        self._model: Any | None = None
        self.model_name = resolved_model
        self.dtype = "fp16" if self.device == "cuda" else "fp32"

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if not allow_hash_fallback:
                raise RuntimeError(
                    "sentence-transformers is required for bge-m3 embeddings. "
                    "Install with `python -m pip install -e .`."
                ) from exc
            self.model_name = FALLBACK_MODEL
            self.dtype = "fp32"
            _log_embedder_init(self.model_name, self.device, self.dtype)
            return

        kwargs: dict[str, Any] = {"device": self.device}
        cache_folder = _cache_folder()
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        self._model = SentenceTransformer(resolved_model, **kwargs)
        if self.device == "cuda":
            self._model.half()
        _log_embedder_init(self.model_name, self.device, self.dtype)

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return 1024

    def encode(self, texts: list[str]) -> Any:
        """Return normalized vectors; keep batch chunk indexing uncached."""

        if not texts:
            return _array([])
        if self._model is not None:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return _array(vectors)
        return _array([_hash_embedding(text, self.dimension) for text in texts])

    def embed_query_cached(self, text: str) -> "np.ndarray":
        """Return a cached query vector as a fresh mutable array."""

        cached = self._embed_query_cached_tuple(text)
        vector = _array(cached)
        if hasattr(vector, "copy"):
            return vector.copy()
        return list(cached)

    @lru_cache(maxsize=256)
    def _embed_query_cached_tuple(self, text: str) -> tuple[float, ...]:
        vectors = self.encode([text])
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return tuple(float(value) for value in vectors[0])


def detect_device() -> str:
    """Return the requested device, otherwise prefer cuda then cpu."""

    configured = os.environ.get("CONTEXT_RAG_DEVICE")
    if configured:
        device = configured.lower()
        if device not in {"cuda", "cpu", "mps"}:
            raise ValueError("CONTEXT_RAG_DEVICE must be one of: cuda, cpu, mps")
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def _hash_embedding(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return normalize(vector)


def _array(vectors: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return vectors
    return np.asarray(vectors, dtype="float32")


def _cache_folder() -> str | None:
    if os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        return os.environ["SENTENCE_TRANSFORMERS_HOME"]
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    return None


def _log_embedder_init(model_name: str, device: str, dtype: str) -> None:
    print(
        f"context-rag: embedder model={model_name} device={device} dtype={dtype}",
        file=sys.stderr,
    )
