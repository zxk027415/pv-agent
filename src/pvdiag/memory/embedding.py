"""Reviewed local BGE embeddings; no hosted endpoint or implicit model download."""

from importlib.metadata import version

import numpy as np

from pvdiag.knowledge.model_assets import verify_snapshot
from pvdiag.knowledge.retrieval import _model_fingerprint


class MemoryEmbedding:
    def __init__(self, path):
        snapshot = verify_snapshot(path, "embedding")
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(path),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"dtype": torch.float32, "use_safetensors": True},
        )
        self.model.max_seq_length = 512
        self.dimensions = self.model.get_sentence_embedding_dimension()
        self.identity = {
            "backend": "reviewed-local-bge-memory-v1",
            "snapshot": snapshot.model_dump(mode="json"),
            "model_sha256": _model_fingerprint(path),
            "dimensions": self.dimensions,
            "query_prefix": "为这个句子生成表示以用于检索相关文章：",
            "max_seq_length": 512,
            "sentence_transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "dtype": "float32",
        }

    def encode(self, text, *, query=False):
        value = self.identity["query_prefix"] + text if query else text
        return self.model.encode(
            [value], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0].tolist()
