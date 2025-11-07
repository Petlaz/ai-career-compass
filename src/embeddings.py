"""Embedding utilities using Sentence-BERT."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

from .utils import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_PATH = DATA_DIR / "embeddings.pkl"
FAISS_INDEX_PATH = DATA_DIR / "embeddings.index"
FAISS_METADATA_PATH = DATA_DIR / "embeddings.index.json"

torch.set_num_threads(4)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and memoize the Sentence-BERT model."""
    logger.info("Loading SentenceTransformer model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME, device="cpu")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def get_embeddings(text_list: list[str]) -> np.ndarray:
    """Return normalized embeddings for the provided text list."""
    if not text_list:
        return np.empty((0, 384))

    model = _load_model()
    embeddings = model.encode(text_list, convert_to_numpy=True, device="cpu", show_progress_bar=False)
    return _normalize(embeddings)


def _digest_texts(texts: Iterable[str]) -> str:
    joined = "||".join(texts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def load_cached_embeddings(
    texts: list[str],
    cache_path: Path = EMBEDDING_CACHE_PATH,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load cached embeddings for the provided texts or compute and cache them."""
    cache_path = Path(cache_path)
    digest = _digest_texts(texts)

    if not force_recompute and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("digest") == digest:
                logger.info("Loaded embeddings from cache at %s", cache_path)
                return payload["embeddings"]
        except (pickle.PickleError, KeyError):
            logger.warning("Failed to load embeddings cache; recomputing.")

    embeddings = get_embeddings(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump({"digest": digest, "embeddings": embeddings}, f)
    logger.info("Cached embeddings for %d texts at %s", len(texts), cache_path)
    return embeddings


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    embeddings32 = embeddings.astype("float32", copy=False)
    index = faiss.IndexFlatIP(embeddings32.shape[1])
    index.add(embeddings32)
    return index


def _load_faiss_index(
    embeddings: np.ndarray,
    digest: str,
    index_path: Path,
    metadata_path: Path,
    force_recompute: bool = False,
) -> faiss.IndexFlatIP:
    if not force_recompute and index_path.exists() and metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("digest") == digest and meta.get("size") == len(embeddings):
                logger.info("Loaded FAISS index from %s", index_path)
                return faiss.read_index(str(index_path))
        except (json.JSONDecodeError, OSError, RuntimeError):
            logger.warning("Failed to load FAISS index metadata; rebuilding.")

    index = _build_faiss_index(embeddings)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({"digest": digest, "size": len(embeddings)}, f)
    logger.info("Rebuilt FAISS index with %d vectors at %s", len(embeddings), index_path)
    return index


def load_embeddings_with_index(
    texts: list[str],
    embedding_cache_path: Path = EMBEDDING_CACHE_PATH,
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = FAISS_METADATA_PATH,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
    embeddings = load_cached_embeddings(
        texts,
        cache_path=embedding_cache_path,
        force_recompute=force_recompute,
    )
    digest = _digest_texts(texts)
    index = _load_faiss_index(
        embeddings,
        digest=digest,
        index_path=index_path,
        metadata_path=metadata_path,
        force_recompute=force_recompute,
    )
    return embeddings, index


__all__ = [
    "get_embeddings",
    "load_cached_embeddings",
    "load_embeddings_with_index",
    "MODEL_NAME",
    "EMBEDDING_CACHE_PATH",
    "FAISS_INDEX_PATH",
    "FAISS_METADATA_PATH",
]
