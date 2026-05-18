"""Local embedding helpers."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Sequence


DEFAULT_MODEL = "BAAI/bge-m3"
FALLBACK_MODEL = "local-hashing-1024"
TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


class Embedder:
    """Encode text with bge-m3 when available, otherwise a local test fallback."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 16,
        device: str | None = None,
        allow_hash_fallback: bool = True,
    ) -> None:
        self.requested_model_name = model_name
        self.batch_size = batch_size
        self.device = device or detect_device()
        self._model: Any | None = None
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if not allow_hash_fallback:
                raise RuntimeError(
                    "sentence-transformers is required for bge-m3 embeddings. "
                    "Install with `python -m pip install -e .`."
                ) from exc
            self.model_name = FALLBACK_MODEL
            return

        kwargs: dict[str, Any] = {"device": self.device}
        cache_folder = _cache_folder()
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        self._model = SentenceTransformer(model_name, **kwargs)

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return 1024

    def encode(self, texts: list[str]) -> Any:
        """Return normalized vectors for the supplied texts."""

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


def detect_device() -> str:
    """Prefer mps, then cuda, then cpu."""

    try:
        import torch
    except ImportError:
        return "cpu"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
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
