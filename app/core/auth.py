from uuid import UUID

from fastapi import HTTPException, Request


async def get_user_id(request: Request) -> UUID:
    """Единая dependency бизнес-кода.

    `user_id` берётся из `request.state`, который заполнила middleware
    внутренней аутентификации ПОСЛЕ проверки X-Internal-Token. Бизнес-код
    не читает X-User-Id напрямую из заголовка.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(401, "Не передан идентификатор пользователя")
    return user_id
