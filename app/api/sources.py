from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_owned_completion

router = APIRouter(
    prefix="/v1/chat/completions/{completion_id}/sources", tags=["sources"])


@router.get("")
async def get_sources(completion_id: str, msg=Depends(get_owned_completion)):
    return {
        "id": completion_id,
        "retrieved": msg.retrieved_chunks or [],
        "used_sources": msg.sources or [],
    }
