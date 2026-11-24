"""Регрессия: импорт приложения не должен грузить локальные модели.

Раньше ``app/core/retrieval.py`` создавал ``BgeM3Embedder``/``Reranker`` и
делал ``QdrantClient.scroll`` на уровне модуля. Из-за этого ``python main.py``
с ``reload=True`` загружал по копии моделей в родителе и в spawn-потомке
uvicorn — две копии на одной карте и ``torch.OutOfMemoryError`` на старте.

Тест намеренно не трогает реальный GPU и не ходит в сеть.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from app.core.config import Settings

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _settings(**overrides) -> Settings:
    data = dict(
        DB_HOST="localhost", DB_PORT=5433, DB_USER="u", DB_PASSWORD="p",
        DB_NAME="db", EMBED_MODEL="BAAI/bge-m3", OLLAMA_MODEL="qwen3:8b",
        MINERU_API_URL="http://127.0.0.1:8010",
        ENV="dev", INTERNAL_AUTH_MODE="off",
    )
    data.update(overrides)
    return Settings(_env_file=None, **data)


def test_blank_embed_device_defaults_to_cpu():
    """Старый .env с `EMBED_DEVICE=` не должен снова выбирать CUDA."""
    assert _settings().EMBED_DEVICE == "cpu"
    assert _settings(EMBED_DEVICE="").EMBED_DEVICE == "cpu"
    assert _settings(EMBED_DEVICE="   ").EMBED_DEVICE == "cpu"
    assert _settings(EMBED_DEVICE="cuda:0").EMBED_DEVICE == "cuda:0"


def test_app_import_does_not_load_models():
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",   # не рискуем реальной картой
        "EMBED_DEVICE": "cpu",
        # Если eager-загрузка вернётся, тест не должен тянуть гигабайты из сети.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    code = "\n".join([
        "import main",
        "from app.core import retrieval",
        "assert retrieval._embed_model is None, "
        "'эмбеддер загрузился при импорте — вернулась eager-загрузка'",
        "assert retrieval._reranker is None, "
        "'реранкер загрузился при импорте — вернулась eager-загрузка'",
        "assert retrieval._glossary is None, "
        "'глоссарий загрузился при импорте — вернулась eager-загрузка'",
        "assert retrieval._qdrant is None, "
        "'Qdrant-клиент создан при импорте — вернулся сетевой side effect'",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Loading weights" not in proc.stderr + proc.stdout
