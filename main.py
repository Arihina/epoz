from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from contextlib import asynccontextmanager

from app.api.chat import router as chat_router
from app.api.responses import router as responses_router
from app.api.feedback import router as feedback_router
from app.api.sources import router as sources_router
from app.api.conversations import router as conversations_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="RAG EPoZ Assistant API",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(responses_router)
app.include_router(feedback_router)
app.include_router(sources_router)
app.include_router(conversations_router)


_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    413: "invalid_request_error",
    415: "invalid_request_error",
    422: "invalid_request_error",
}


def _error_body(status_code: int, message: str) -> dict:
    return {"error": {
        "message": message,
        "type": _ERROR_TYPES.get(status_code, "server_error"),
        "param": None,
        "code": None,
    }}


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
    return JSONResponse(status_code=422, content=_error_body(422, message))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        timeout_keep_alive=300,
    )
