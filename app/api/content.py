from __future__ import annotations

from typing import Iterable


CHAT_TEXT_TYPES = frozenset({"text"})

RESPONSES_TEXT_TYPES = frozenset({"input_text", "output_text"})


def extract_text(content, text_types: Iterable[str]) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    types = frozenset(text_types)
    return "\n".join(
        str(p.get("text", ""))
        for p in content
        if isinstance(p, dict) and p.get("type") in types
    )
