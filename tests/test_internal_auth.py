"""Контрактные тесты внутренней аутентификации master -> агент.

Проверяют матрицу из master_node/docs/agent-side-changes.md (раздел 6.7):
- без X-Internal-Token -> 401;
- неверный токен -> 401;
- валидный токен + валидный X-User-Id -> бизнес-ответ;
- валидный токен + отсутствующий/битый X-User-Id -> 401;
- GET /health без токена -> 200;
- /docs, /openapi.json -> 404 в prod-конфиге.

Бизнес-ручки здесь не дёргаются (им нужна БД), поэтому для проверки прохода
валидного запроса используется лёгкий probe-роут, читающий user_id из
request.state — тот же путь, что у реальных ручек через get_user_id.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import ValidationError

from app.core.config import Settings
from app.core.platform_security import install_internal_auth
from main import create_app


TOKEN = "a" * 64
USER_ID = str(uuid4())


def make_settings(*, mode="token", token=TOKEN, public_paths=("/health",),
                  is_prod=False) -> SimpleNamespace:
    return SimpleNamespace(
        ENV="prod" if is_prod else "dev",
        INTERNAL_AUTH_MODE=mode,
        MASTER_INTERNAL_TOKEN=token,
        public_paths=public_paths,
        AGENT_ID="epoz-test",
        is_prod=is_prod,
    )


def build_probe_app(s) -> FastAPI:
    """Минимальное приложение: health + probe, без БД/LLM."""
    app = FastAPI()
    install_internal_auth(app, s)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/probe")
    async def probe(request: Request):
        uid = getattr(request.state, "user_id", None)
        return {"user_id": str(uid)}

    return app


def _get(app, path, *, token=None, user_id=None) -> httpx.Response:
    async def run():
        headers = {}
        if token is not None:
            headers["X-Internal-Token"] = token
        if user_id is not None:
            headers["X-User-Id"] = user_id
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            return await client.get(path, headers=headers)

    return asyncio.run(run())


def test_no_token_401():
    app = build_probe_app(make_settings())
    resp = _get(app, "/probe")
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_wrong_token_401():
    app = build_probe_app(make_settings())
    resp = _get(app, "/probe", token="b" * 64)
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_valid_token_and_user_200():
    app = build_probe_app(make_settings())
    resp = _get(app, "/probe", token=TOKEN, user_id=USER_ID)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == USER_ID


def test_valid_token_missing_user_id_401():
    app = build_probe_app(make_settings())
    resp = _get(app, "/probe", token=TOKEN)
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_valid_token_invalid_user_id_401():
    app = build_probe_app(make_settings())
    resp = _get(app, "/probe", token=TOKEN, user_id="not-a-uuid")
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_health_public_200():
    app = build_probe_app(make_settings())
    resp = _get(app, "/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_request_id_echoed():
    app = build_probe_app(make_settings())
    resp = _get(app, "/health")
    assert resp.headers.get("x-request-id")


def test_off_mode_allows_without_token():
    app = build_probe_app(make_settings(mode="off", token=None))
    resp = _get(app, "/probe", user_id=USER_ID)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == USER_ID


def test_docs_disabled_in_prod():
    app = create_app(make_settings(is_prod=True))
    for path in ("/docs", "/openapi.json", "/redoc"):
        # С валидным внутренним токеном: маршрутов документации в prod нет,
        # поэтому FastAPI отдаёт 404 (а не страницу/схему).
        resp = _get(app, path, token=TOKEN, user_id=USER_ID)
        assert resp.status_code == 404, path


def test_health_through_create_app_200():
    app = create_app(make_settings(is_prod=True))
    resp = _get(app, "/health")
    assert resp.status_code == 200


# --- fail-closed конфигурация (раздел 6.1) ---------------------------------


def _full_settings(**overrides) -> Settings:
    base = dict(
        DB_HOST="localhost", DB_PORT=5433, DB_USER="u", DB_PASSWORD="p",
        DB_NAME="db", EMBED_MODEL="BAAI/bge-m3", OLLAMA_MODEL="qwen3:8b",
        MINERU_API_URL="http://127.0.0.1:8010",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_config_token_without_secret_fails():
    with pytest.raises(ValidationError):
        _full_settings(INTERNAL_AUTH_MODE="token", MASTER_INTERNAL_TOKEN=None)


def test_config_off_in_prod_fails():
    with pytest.raises(ValidationError):
        _full_settings(ENV="prod", INTERNAL_AUTH_MODE="off",
                       MASTER_INTERNAL_TOKEN=None)


def test_config_short_token_in_prod_fails():
    with pytest.raises(ValidationError):
        _full_settings(ENV="prod", INTERNAL_AUTH_MODE="token",
                       MASTER_INTERNAL_TOKEN="short")
