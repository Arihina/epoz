from __future__ import annotations

from uuid import UUID

import pypandoc
from fastapi import APIRouter, Body, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.database import get_db
from app.db import crud
from app.models.models import GeneratedDocument

from app.core.auth import get_user_id
from app.core.context import ContextOverflow
from app.core.llm import generate_document_markdown
from app.api.deps import get_owned_generated_document, parse_document_id, parse_file_id

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_EXT = {"docx": ".docx", "md": ".md"}


def _fmt_doc(d: GeneratedDocument) -> dict:
    return {
        "id": f"doc-{d.id}",
        "object": "document",
        "filename": d.filename,
        "format": d.format,
        "template_file_id": f"file-{d.template_file_id}" if d.template_file_id else None,
        "bytes": d.size_bytes,
        "created_at": int(d.created_at.timestamp()),
    }


def _output_filename(template_filename: str, fmt: str) -> str:
    stem = template_filename.rsplit(".", 1)[0] if "." in template_filename else template_filename
    return stem + _EXT[fmt]


def _to_docx(markdown: str) -> bytes:
    return pypandoc.convert_text(markdown, "docx", format="md")


@router.post("/generate", status_code=201)
async def generate_document(
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

    template_md = template.markdown_content or ""
    if not template_md.strip():
        raise HTTPException(400, "Шаблон пуст")

    try:
        filled = await run_in_threadpool(
            generate_document_markdown, template_md, template.filename,
            instructions, None,
        )
    except ContextOverflow as e:
        raise HTTPException(413, str(e)) from e

    artifact: bytes | None = None
    if fmt == "docx":
        try:
            artifact = await run_in_threadpool(_to_docx, filled)
        except Exception as e:
            raise HTTPException(
                500, f"Не удалось собрать DOCX (pandoc недоступен?): {e}") from e

    size = len(artifact) if artifact is not None else len(filled.encode("utf-8"))
    filename = _output_filename(template.filename, fmt)

    doc = await crud.create_generated_document(
        db, user_id=user_id, template_file_id=template.id,
        filename=filename, format=fmt, markdown_content=filled,
        artifact_bytes=artifact, size_bytes=size,
    )
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
