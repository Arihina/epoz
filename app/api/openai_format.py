from __future__ import annotations


_FILE_STATUS_MAP = {
    "pending": "uploaded",
    "processing": "uploaded",
    "done": "processed",
    "failed": "error",
}


def file_status(internal_status: str) -> str:
    return _FILE_STATUS_MAP.get(internal_status, "error")


def file_object(f) -> dict:
    return {
        "id": f"file-{f.id}",
        "object": "file",
        "bytes": f.size_bytes,
        "created_at": int(f.created_at.timestamp()),
        "filename": f.filename,
        "purpose": "assistants",
        "status": file_status(f.status),
        "status_details": f.error_message,
        # Расширения платформы.
        "processing_status": f.status,
        "conversation_id": str(f.conversation_id) if f.conversation_id else None,
    }
