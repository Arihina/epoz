from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router, recover_stuck_generations
from app.api.feedback import router as feedback_router
from app.api.files import router as files_router, recover_stuck_files
from app.api.responses import router as responses_router
from app.api.sources import router as sources_router
from app.core.config import settings
from app.core.platform_security import install_internal_auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await recover_stuck_files()
    except Exception as exc:
        print(f"[files] reaper при старте не отработал: {exc}")
    try:
        await recover_stuck_generations()
    except Exception as exc:
        print(f"[documents] reaper при старте не отработал: {exc}")
    yield


_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    413: "invalid_request_error",
    415: "invalid_request_error",
    422: "invalid_request_error",
}

_VALIDATION_STATUS = 400

_LOC_PREFIXES = {"body", "query", "path", "header", "cookie"}


def _param_from_loc(loc) -> str | None:
    """('body', 'messages', 0, 'role') -> 'messages.0.role'"""
    parts = [str(p) for p in loc if p not in _LOC_PREFIXES]
    return ".".join(parts) or None


def _error_body(status_code: int, message: str, param: str | None = None) -> dict:
    return {"error": {
        "message": message,
        "type": _ERROR_TYPES.get(status_code, "server_error"),
        "param": param,
        "code": None,
    }}


def create_app(app_settings=None) -> FastAPI:
    if app_settings is None:
        app_settings = settings

    docs_on = not app_settings.is_prod

    app = FastAPI(
        title="RAG EPoZ Assistant API",
        version="3.0.0",
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        openapi_url="/openapi.json" if docs_on else None,
        lifespan=lifespan,
    )

    install_internal_auth(app, app_settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(responses_router)
    app.include_router(feedback_router)
    app.include_router(sources_router)
    app.include_router(conversations_router)
    app.include_router(files_router)
    app.include_router(documents_router)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.status_code, str(exc.detail)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        message = first.get("msg", "Некорректный запрос")
        param = _param_from_loc(first.get("loc", ()))

        return JSONResponse(
            status_code=_VALIDATION_STATUS,
            content=_error_body(_VALIDATION_STATUS, message, param),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=not settings.is_prod,
        timeout_keep_alive=300,
    )
