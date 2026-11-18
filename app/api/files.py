from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File as FastAPIFile, Form, HTTPException, UploadFile
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal, get_db
from app.db import crud
from app.models.models import File

from app.core import mineru
from app.core.auth import get_user_id
from app.api.deps import get_owned_file
from app.api.openai_format import file_object as _fmt

router = APIRouter(prefix="/v1/files", tags=["files"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_MINERU_CONCURRENCY = 2

_mineru_semaphore = asyncio.Semaphore(MAX_MINERU_CONCURRENCY)
_bg_tasks: set[asyncio.Task] = set()


def _ocr_backend() -> str:
    return f"mineru:{mineru.settings.MINERU_BACKEND}:{mineru.settings.MINERU_LANG}"


async def process_file(file_id: UUID) -> None:
    """Фоновая обработка: pending -> processing -> done/failed.

    Отдельная сессия — request-сессия к моменту запуска BackgroundTasks уже
    закрыта. Семафор ограничивает число одновременных вызовов MinerU.
    """
    async with _mineru_semaphore:
        async with AsyncSessionLocal() as db:
            f = await crud.get_file_content(db, file_id)
            if f is None or f.status != "pending":
                return
            content = f.content
            await crud.set_file_processing(db, f)
            try:
                markdown = await mineru.parse_document(
                    filename=f.filename, content=content, mime_type=f.mime_type,
                )
            except mineru.MinerUError as e:
                await crud.set_file_failed(db, f, str(e))
                return
            await crud.set_file_done(
                db, f, markdown_content=markdown,
                ocr_backend=_ocr_backend(), markdown_tokens=None,
            )


def _spawn(file_id: UUID) -> None:
    task = asyncio.create_task(process_file(file_id))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def recover_stuck_files() -> None:
    """Reaper: на старте вернуть зависшие processing в pending и перезапустить
    все pending — их фоновая задача была потеряна при прошлом рестарте."""
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(File)
            .where(File.status == "processing")
            .values(status="pending")
        )
        result = await db.execute(select(File.id).where(File.status == "pending"))
        ids = [row[0] for row in result.all()]
        await db.commit()

    for file_id in ids:
        _spawn(file_id)


@router.post("", status_code=201)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = FastAPIFile(...),
    conversation_id: str | None = Form(None),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv_uuid: UUID | None = None
    if conversation_id is not None:
        try:
            conv_uuid = UUID(conversation_id)
        except ValueError:
            raise HTTPException(400, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conv_uuid, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Файл больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ")

    f = await crud.create_file(
        db, user_id=user_id,
        filename=file.filename or "документ",
        mime_type=file.content_type,
        size_bytes=len(content),
        content=content,
        conversation_id=conv_uuid,
    )

    background_tasks.add_task(process_file, f.id)
    return _fmt(f)


@router.get("")
async def list_files(
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    return {"object": "list", "data": [_fmt(f) for f in await crud.list_files(db, user_id)]}


@router.get("/{file_id}")
async def get_file(f: File = Depends(get_owned_file)):
    return _fmt(f)


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    f: File = Depends(get_owned_file),
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_file(db, f)
    return {"id": f"file-{f.id}", "object": "file", "deleted": True}
