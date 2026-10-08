from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class RerankBackend(Protocol):
    name: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class HashEmbeddingBackend:
    """Small deterministic backend for unit tests; never used for real evaluation."""

    name = "test-hash-v1"

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def _one(self, text: str) -> np.ndarray:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = bytes(seed[index % len(seed)] for index in range(self.dimensions))
        vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5
        norm = float(np.linalg.norm(vector)) or 1.0
        return vector / norm

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack([self._one(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._one("query:" + text)


class KeywordReranker:
    """Deterministic test reranker based on shared character bigrams."""

    name = "test-keyword-v1"

    @staticmethod
    def _grams(text: str) -> set[str]:
        compact = "".join(text.casefold().split())
        return {compact[i : i + 2] for i in range(max(0, len(compact) - 1))}

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_grams = self._grams(query)
        return [float(len(query_grams & self._grams(document))) for document in documents]


class BGEEmbeddingBackend:
    name = "bge-small-zh-v1.5-pytorch"

    def __init__(self, model_path: Path) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.functional = functional
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, dtype=self.dtype
        ).to(self.device).eval()

    def _encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**inputs, return_dict=True).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normalized = self.functional.normalize(pooled.float(), p=2, dim=1)
            batches.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(["为这个句子生成表示以用于检索相关文章：" + text], batch_size=1)[0]


class BGEReranker:
    name = "bge-reranker-base"

    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, dtype=self.dtype
        ).to(self.device).eval()

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        inputs = self.tokenizer(
            [[query, document] for document in documents],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu()
        return scores.tolist()
