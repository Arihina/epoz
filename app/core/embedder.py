"""BGE-M3: dense-векторы и разреженные (lexical) веса за один проход модели.

Единая точка загрузки модели для индексирования (create_collection_hybrid.py)
и поиска (app/core/retrieval.py) — веса и токенизатор одни и те же, поэтому
dense/sparse из индекса и из запроса сопоставимы.

Используется официальная обёртка FlagEmbedding (BGEM3FlagModel): она отдаёт
dense (CLS-пулинг + L2-нормализация) и lexical_weights — словарь
``token_id -> вес``, где вес — результат ``relu(sparse_linear(hidden))``,
схлопнутый по повторяющимся токенам через max.

BGE-M3 не требует префиксов ``query:``/``passage:`` — текст эмбеддится как есть.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from huggingface_hub import snapshot_download

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

DENSE_DIM = 1024

DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 64


def resolve_model_path(model_name: str, local_files_only: bool) -> str:
    """Локальный путь к снапшоту модели в кэше HF (для offline-режима)."""
    if not local_files_only:
        return model_name
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Модель {model_name} не найдена в локальном кэше HuggingFace. "
            f"Скачайте её заранее: huggingface-cli download {model_name}"
        ) from exc


def pick_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class BgeM3Embedder:
    """Обёртка над BGE-M3: dense + sparse одним вызовом encode()."""

    def __init__(
        self,
        model_name: str,
        *,
        local_files_only: bool = False,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_path = resolve_model_path(model_name, local_files_only)
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = BGEM3FlagModel(
            self.model_path,
            use_fp16=self.device.startswith("cuda"),
            devices=self.device,
            batch_size=batch_size,
            query_max_length=max_length,
            passage_max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        """Вернуть (dense-матрица [n, 1024], список словарей token_id -> вес)."""
        result = self.model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            max_length=max_length or self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = np.asarray(result["dense_vecs"], dtype=np.float32)
        if dense.ndim == 1:
            dense = dense[None, :]

        lexical = result["lexical_weights"] or []
        if isinstance(lexical, dict):
            lexical = [lexical]
        sparse = [self._to_sparse(weights) for weights in lexical]

        return dense, sparse

    @staticmethod
    def _to_sparse(weights: dict[str, float]) -> dict[int, float]:
        """Ключи lexical_weights приходят строками и уже отфильтрованы по w > 0."""
        return {
            int(token_id): float(weight)
            for token_id, weight in weights.items()
            if float(weight) > 0
        }


class Reranker:
    """Cross-encoder для переранжирования кандидатов (bge-reranker-v2-m3)."""

    def __init__(
        self,
        model_name: str,
        *,
        local_files_only: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_path = resolve_model_path(model_name, local_files_only)
        self.device = device or pick_device()
        self.model = FlagReranker(
            self.model_path,
            use_fp16=self.device.startswith("cuda"),
            devices=self.device,
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Релевантность каждой пары (query, text) — список скоуров той же длины."""
        if not texts:
            return []
        scores = self.model.compute_score([[query, t] for t in texts])
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]
