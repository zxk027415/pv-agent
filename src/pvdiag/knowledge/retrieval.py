"""Deterministic BM25, reciprocal-rank fusion and optional local BGE reranking."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np


def tokens(text):
    text = text.casefold()
    words = re.findall(r"[a-z0-9_]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run)
        words.extend(run[i : i + 2] for i in range(len(run) - 1))
    return words


def bm25(query, documents):
    bags = [Counter(tokens(doc)) for doc in documents]
    lengths = [sum(bag.values()) for bag in bags]
    average = sum(lengths) / max(1, len(lengths))
    scores = [0.0] * len(bags)
    # Stable summation order is required for identical scores/evidence IDs across processes.
    for term in sorted(set(tokens(query))):
        count = sum(term in bag for bag in bags)
        weight = math.log(1 + (len(bags) - count + 0.5) / (count + 0.5))
        for i, bag in enumerate(bags):
            frequency = bag[term]
            if frequency:
                normalizer = 1.5 * (0.25 + 0.75 * lengths[i] / average)
                scores[i] += weight * frequency * 2.5 / (frequency + normalizer)
    return scores


def _matrix(values, size):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != size or array.shape[1] < 1:
        raise ValueError("invalid embedding dimensions")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(array).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("embeddings must be finite nonzero vectors")
    return array / norms[:, None]


class Retriever:
    def __init__(self, texts, *, mode="bm25", neural=None):
        if mode not in {"bm25", "hybrid_bge"}:
            raise ValueError("unknown retrieval mode")
        if mode == "hybrid_bge" and neural is None:
            raise ValueError("hybrid_bge requires local embedding and reranker models")
        self.texts, self.mode, self.neural = list(texts), mode, neural
        self.config = {
            "mode": mode,
            "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "latin_cjk_bigrams_v1"},
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if mode == "hybrid_bge":
            self.config.update({"rrf_k": 60, "candidates": 20, "neural": neural.identity})
            self.vectors = _matrix(neural.encode(self.texts, query=False), len(texts))

    def search(self, query, eligible, top_k):
        lexical = bm25(query, self.texts)
        lexical_rank = sorted(
            (i for i in eligible if lexical[i] > 0), key=lambda i: (-lexical[i], i)
        )
        if self.mode == "bm25":
            return [(i, lexical[i]) for i in lexical_rank[:top_k]]
        query_vector = _matrix(self.neural.encode([query], query=True), 1)
        if query_vector.shape[1] != self.vectors.shape[1]:
            raise ValueError("query and corpus embedding dimensions differ")
        similarities = self.vectors @ query_vector[0]
        dense_rank = sorted(eligible, key=lambda i: (-similarities[i], i))
        fused = {}
        for ranking in (lexical_rank[:20], dense_rank[:20]):
            for rank, i in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        candidates = sorted(fused, key=lambda i: (-fused[i], i))[:20]
        if not candidates:
            return []
        scores = np.asarray(
            self.neural.rerank(query, [self.texts[i] for i in candidates]), dtype=float
        )
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("invalid reranker scores")
        ranked = sorted(zip(candidates, scores.tolist()), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]


def _model_fingerprint(path):
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("local model path must be a directory")
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*")):
        if not file.is_file() or ".cache" in file.relative_to(root).parts:
            continue
        digest.update(file.relative_to(root).as_posix().encode() + b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class LocalBGE:
    """Explicit local models only; never download or use hosted embeddings implicitly.

    Requires sentence-transformers, not part of the current installed core runtime.
    Model revisions are identified by complete local artifact hashes, not aliases.
    """

    def __init__(self, embedding_path, reranker_path, *, device="cpu"):
        if not embedding_path or not reranker_path:
            raise ValueError("both local model directories are required")
        self.identity = {
            "backend": "sentence-transformers-local",
            "embedding_sha256": _model_fingerprint(embedding_path),
            "reranker_sha256": _model_fingerprint(reranker_path),
            "query_prefix": "为这个句子生成表示以用于检索相关文章：",
            "device": device,
        }
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "hybrid_bge requires the optional sentence-transformers runtime"
            ) from None
        self.identity["sentence_transformers_version"] = version("sentence-transformers")
        self.embedding = SentenceTransformer(
            str(embedding_path), device=device, local_files_only=True, trust_remote_code=False
        )
        self.reranker = CrossEncoder(
            str(reranker_path), device=device, local_files_only=True, trust_remote_code=False
        )

    def encode(self, texts, *, query):
        if query:
            texts = [self.identity["query_prefix"] + text for text in texts]
        return self.embedding.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def rerank(self, query, texts):
        return self.reranker.predict([(query, text) for text in texts], show_progress_bar=False)
