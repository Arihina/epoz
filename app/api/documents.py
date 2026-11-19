from __future__ import annotations

import asyncio
from uuid import UUID

import pypandoc
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.database import AsyncSessionLocal, get_db
from app.db import crud
from app.models.models import GeneratedDocument

from app.core.auth import get_user_id
from app.core.context import ContextOverflow
from app.core.llm import generate_document_markdown
from app.api.deps import get_owned_generated_document, parse_document_id, parse_file_id

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_EXT = {"docx": ".docx", "md": ".md"}
MAX_GENERATION_CONCURRENCY = 2

_gen_semaphore = asyncio.Semaphore(MAX_GENERATION_CONCURRENCY)
_bg_tasks: set[asyncio.Task] = set()


def _fmt_doc(d: GeneratedDocument) -> dict:
    return {
        "id": f"doc-{d.id}",
        "object": "document",
        "filename": d.filename,
        "format": d.format,
        "template_file_id": f"file-{d.template_file_id}" if d.template_file_id else None,
        "bytes": d.size_bytes,
        "status": d.status,
        "status_details": d.error_message,
        "created_at": int(d.created_at.timestamp()),
    }


def _output_filename(template_filename: str, fmt: str) -> str:
    stem = template_filename.rsplit(".", 1)[0] if "." in template_filename else template_filename
    return stem + _EXT[fmt]


def _to_docx(markdown: str) -> bytes:
    return pypandoc.convert_text(markdown, "docx", format="md")


async def process_generation(doc_id: UUID) -> None:
    """Фоновая генерация: pending -> processing -> done/failed."""
    async with _gen_semaphore:
        async with AsyncSessionLocal() as db:
            doc = await crud.get_generated_document(db, doc_id)
            if doc is None or doc.status != "pending":
                return

            template = None
            if doc.template_file_id is not None:
                template = await crud.get_file_for_user(
                    db, doc.template_file_id, doc.user_id)
            if template is None or template.status != "done":
                await crud.set_generated_document_failed(
                    db, doc, "Шаблон не найден или не готов")
                return

            template_md = template.markdown_content or ""
            await crud.set_generated_document_processing(db, doc)

            try:
                filled = await run_in_threadpool(
                    generate_document_markdown, template_md, template.filename,
                    doc.instructions, None,
                )
            except ContextOverflow as e:
                await crud.set_generated_document_failed(db, doc, str(e))
                return
            except Exception as e:
                await crud.set_generated_document_failed(db, doc, str(e))
                return

            artifact: bytes | None = None
            if doc.format == "docx":
                try:
                    artifact = await run_in_threadpool(_to_docx, filled)
                except Exception as e:
                    await crud.set_generated_document_failed(
                        db, doc, f"Не удалось собрать DOCX (pandoc): {e}")
                    return

            size = len(artifact) if artifact is not None else len(filled.encode("utf-8"))
            await crud.set_generated_document_done(
                db, doc, markdown_content=filled,
                artifact_bytes=artifact, size_bytes=size,
            )


def _spawn(doc_id: UUID) -> None:
    task = asyncio.create_task(process_generation(doc_id))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def recover_stuck_generations() -> None:
    """Reaper: зависшие processing -> pending, перезапуск pending."""
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(GeneratedDocument)
            .where(GeneratedDocument.status == "processing")
            .values(status="pending")
        )
        result = await db.execute(
            select(GeneratedDocument.id).where(GeneratedDocument.status == "pending")
        )
        ids = [row[0] for row in result.all()]
        await db.commit()

    for doc_id in ids:
        _spawn(doc_id)


@router.post("/generate", status_code=201)
async def generate_document(
    background_tasks: BackgroundTasks,
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    template_file_id = body.get("template_file_id")
    instructions = body.get("instructions")
    fmt = body.get("format", "docx")

    if fmt not in _EXT:
        raise HTTPException(400, "format должен быть 'docx' или 'md'")
    if not template_file_id:
        raise HTTPException(400, "template_file_id обязателен")

    tid = parse_file_id(template_file_id)
    template = await crud.get_file_for_user(db, tid, user_id)
    if template is None:
        raise HTTPException(404, "Шаблон (template_file_id) не найден")
    if template.status != "done":
        raise HTTPException(
            400, f"Шаблон ещё не готов (статус обработки: {template.status})")

    filename = _output_filename(template.filename, fmt)
    doc = await crud.create_generated_document(
        db, user_id=user_id, template_file_id=template.id,
        filename=filename, format=fmt, instructions=instructions,
    )

    background_tasks.add_task(process_generation, doc.id)
    return _fmt_doc(doc)


@router.get("")
async def list_documents(
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    return {"object": "list", "data": [_fmt_doc(d) for d in await crud.list_generated_documents(db, user_id)]}


@router.get("/{document_id}")
async def get_document(d: GeneratedDocument = Depends(get_owned_generated_document)):
    return _fmt_doc(d)


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    d = await crud.get_generated_document_artifact(
        db, parse_document_id(document_id), user_id)
    if d is None:
        raise HTTPException(404, "Документ не найден")

    if d.format == "docx":
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        body = d.artifact_bytes or b""
    else:
        media = "text/markdown"
        body = d.markdown_content.encode("utf-8")

    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{d.filename}"'},
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    d: GeneratedDocument = Depends(get_owned_generated_document),
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_generated_document(db, d)
    return {"id": f"doc-{d.id}", "object": "document", "deleted": True}
