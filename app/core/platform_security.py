"""Внутренняя аутентификация master -> агент и сквозная корреляция логов.

Это НЕ пользовательская аутентификация: здесь агент убеждается, что запрос
пришёл от master (per-agent секрет `X-Internal-Token`), и принимает
проверенный master'ом `X-User-Id`. Пользовательский JWT Keycloak агент не
видит и в Keycloak не ходит.

Реализовано чистыми ASGI-middleware, без `BaseHTTPMiddleware`: epoz отдаёт
SSE-стримы (chat/responses), а `BaseHTTPMiddleware` исторически их ломает.

Порядок middleware (outermost -> innermost):
    RequestContextMiddleware -> InternalAuthMiddleware -> CORS -> router

`RequestContextMiddleware` самый внешний: `request_id` доступен даже на
401 от `InternalAuthMiddleware`, а access-лог пишется для каждого HTTP-запроса.
"""

from __future__ import annotations

import hmac
import logging
import re
import time
import uuid
from uuid import UUID

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Заголовок, которым master доказывает агенту своё авторство.
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
# Пользователь из проверенного JWT/sub (уже провалидирован master).
USER_ID_HEADER = "X-User-Id"
# Сквозной идентификатор запроса. Не является аутентификацией.
REQUEST_ID_HEADER = "X-Request-Id"
# Этап 2: подписанный assertion вместо статического токена (не реализован).
ASSERTION_HEADER = "X-Master-Assertion"

MIN_PROD_TOKEN_LENGTH = 32

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

logger = logging.getLogger("epoz.internal_auth")


def new_request_id() -> str:
    return uuid.uuid4().hex


def normalize_request_id(value: str | None) -> str:
    """Принять валидный X-Request-Id или сгенерировать новый.

    Некорректный id — не повод отдавать 400: это диагностический заголовок,
    а не часть аутентификации. Но пропускать его в логи как есть нельзя.
    """
    if value and _REQUEST_ID_RE.match(value):
        return value
    return new_request_id()


def error_body(
    message: str,
    error_type: str = "authentication_error",
    param: str | None = None,
) -> dict:
    """Единый формат ошибок платформы (совпадает с master)."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
    }


def _safe_log_value(value: object, limit: int = 512) -> str:
    """Убрать из значения для лога управляющие символы.

    `scope["path"]` в ASGI уже URL-декодирован, поэтому `%0a` в URL мог бы
    превратиться в перевод строки и подделать следующую запись лога.
    """
    text = str(value)[:limit]
    return "".join(ch if ch.isprintable() else "?" for ch in text)


class InternalAuthMiddleware:
    """Проверка X-Internal-Token и валидация X-User-Id как UUID.

    Успешный результат кладёт `user_id` в `scope["state"]["user_id"]`, откуда
    его читает dependency `get_user_id` (app/core/auth.py). Бизнес-код не
    читает заголовки напрямую.
    """

    def __init__(self, app: ASGIApp, settings) -> None:
        self.app = app
        self.mode = settings.INTERNAL_AUTH_MODE
        self.expected_token = settings.MASTER_INTERNAL_TOKEN
        # Точное совпадение путей, без startswith.
        self.public_paths = frozenset(settings.public_paths)

    async def __call__(self, scope: Scope, receive: Receive,
                       send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.public_paths:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)

        if self.mode == "off":
            # Локальная разработка/тесты: токен не проверяем, но user_id
            # по-прежнему берём из заголовка и валидируем как UUID (старый
            # контракт dev_header). Режим должен явно логировать предупреждение
            # (см. install_internal_auth).
            parsed = self._parse_user_id(headers)
            if parsed is None:
                await self._reject(
                    scope, receive, send,
                    "Некорректный идентификатор пользователя",
                )
                return
            scope.setdefault("state", {})["user_id"] = parsed
            await self.app(scope, receive, send)
            return

        if self.mode == "token":
            token = headers.get(INTERNAL_TOKEN_HEADER)
            # hmac.compare_digest без раннего выхода по длине.
            if not token or not hmac.compare_digest(
                token, self.expected_token or ""
            ):
                await self._reject(
                    scope, receive, send, "Требуется внутренний токен master"
                )
                return
        elif self.mode == "assertion":
            # Этап 2 (подписанный assertion) в epoz ещё не реализован.
            # Fail-closed: не пропускаем запрос без проверенной подписи.
            await self._reject(
                scope, receive, send, "Требуется внутренний токен master"
            )
            return

        parsed = self._parse_user_id(headers)
        if parsed is None:
            await self._reject(
                scope, receive, send,
                "Некорректный идентификатор пользователя",
            )
            return

        scope.setdefault("state", {})["user_id"] = parsed
        await self.app(scope, receive, send)

    @staticmethod
    def _parse_user_id(headers: Headers) -> UUID | None:
        try:
            return UUID(headers.get(USER_ID_HEADER))
        except (ValueError, TypeError):
            return None

    async def _reject(self, scope: Scope, receive: Receive, send: Send,
                      message: str) -> None:
        # Возвращаем ответ напрямую, а не raise HTTPException: пользовательские
        # middleware в Starlette выполняются снаружи ExceptionMiddleware, и
        # исключение здесь превратилось бы в 500.
        response = JSONResponse(
            status_code=401,
            content=error_body(message, "authentication_error"),
        )
        await response(scope, receive, send)


class RequestContextMiddleware:
    """Чистый ASGI-middleware: request_id + access-лог."""

    def __init__(self, app: ASGIApp, settings) -> None:
        self.app = app
        self.agent_id = settings.AGENT_ID

    async def __call__(self, scope: Scope, receive: Receive,
                       send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = normalize_request_id(incoming)
        # request.state читает именно scope["state"]; middleware выполняется
        # до Router, поэтому словарь создаём сами.
        scope.setdefault("state", {})["request_id"] = request_id

        status = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            user_id = scope.get("state", {}).get("user_id")
            logger.info(
                "request_id=%s agent_id=%s method=%s path=%s status=%d "
                "user_id=%s duration_ms=%.1f",
                request_id,
                self.agent_id,
                _safe_log_value(scope.get("method", "")),
                _safe_log_value(scope.get("path", "")),
                status,
                user_id or "-",
                duration_ms,
            )


def install_internal_auth(app: ASGIApp, settings) -> None:
    """Подключить middleware внутренней аутентификации до включения роутеров."""
    # Порядок add_middleware важен: первый вызов — самый внешний слой.
    app.add_middleware(RequestContextMiddleware, settings=settings)
    app.add_middleware(InternalAuthMiddleware, settings=settings)

    if settings.INTERNAL_AUTH_MODE == "off":
        logger.warning(
            "INTERNAL_AUTH_MODE=off: внутренняя аутентификация отключена "
            "(только для локальной разработки/тестов; верните token)"
        )
