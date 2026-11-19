from __future__ import annotations

from dataclasses import dataclass, field

from app.core.tokens import TokenCounter


JOIN_SLACK = 16

HEAD_RAG = """Ты русскоязычный AI ассистент.
Отвечай ТОЛЬКО на русском языке.
Отвечай СТРОГО на основании приведённого контекста.
Если в контексте нет ответа на вопрос — прямо напиши, что в положении такая информация не найдена, и НЕ додумывай.
Не выдумывай нормы, сроки и критерии, которых нет в контексте."""

TRUNCATION_NOTE = (
    "[Документ приведён частично: помещено примерно {percent}% содержимого, "
    "остальное не влезло в контекст модели. Если ответа в приведённой части "
    "нет — прямо скажи, что документ показан не полностью, и не делай выводов "
    "о том, чего в нём нет.]"
)

PARTIAL_SUFFIX = " (частично)"


class ContextOverflow(Exception):
    """Промпт не помещается в контекстное окно модели.

    Транспортный слой переводит это в 413. Модуль намеренно не знает про
    FastAPI.
    """

    def __init__(self, message: str, needed: int, available: int) -> None:
        super().__init__(message)
        self.needed = needed
        self.available = available


@dataclass
class Document:
    filename: str
    markdown: str
    tokens: int | None = None


@dataclass
class PromptStats:
    num_ctx: int
    prompt_tokens: int
    reserved_output: int
    documents: int
    truncated: int
    document_tokens: int
    history_kept: int
    history_total: int
    history_budget: int
    counter: str

    def as_log(self) -> str:
        trunc = f" ({self.truncated} обрезан)" if self.truncated else ""
        return (
            f"ctx={self.num_ctx} prompt≈{self.prompt_tokens} "
            f"reserve={self.reserved_output} docs={self.documents}{trunc} "
            f"doc_tokens={self.document_tokens} "
            f"history={self.history_kept}/{self.history_total} "
            f"history_budget={self.history_budget} counter={self.counter}"
        )


@dataclass
class BuiltPrompt:
    prompt: str
    sources: list[str] | None
    stats: PromptStats
    truncated_files: list[str] = field(default_factory=list)


def _fit_shares(costs: list[int], budget: int) -> list[int]:
    """Равные доли с перераспределением излишков: маленькие чанки помещаются
    целиком, а освободившийся остаток достаётся большим документам."""
    alloc = [0] * len(costs)
    hungry = list(range(len(costs)))
    left = budget

    while hungry and left > 0:
        share = left // len(hungry)
        if share == 0:
            break
        still: list[int] = []
        for i in hungry:
            take = min(share, costs[i] - alloc[i])
            alloc[i] += take
            left -= take
            if alloc[i] < costs[i]:
                still.append(i)
        hungry = still

    return alloc


def _truncate(text: str, limit: int, counter: TokenCounter) -> str:
    """Обрезать текст до limit токенов по границе markdown-блока."""
    if limit <= 0:
        return ""

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1

    cut = lo
    boundary = text.rfind("\n\n", 0, cut)
    if boundary > cut * 0.6:
        cut = boundary

    return text[:cut].rstrip()


def _history_block(history: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{'Пользователь' if role == 'user' else 'Ассистент'}: {text}"
        for role, text in history
    )


def _skeleton(head, instructions, context_block, attached_block, history, question) -> str:
    instr = ""
    if instructions and instructions.strip():
        instr = f"\nДополнительные инструкции пользователя:\n{instructions.strip()}\n"
    doc_section = f"{attached_block}\n" if attached_block else ""
    return f"""{head}
{instr}
Контекст:
{context_block}
{doc_section}
Предыдущий диалог:
{history}

Вопрос пользователя:
{question}

Ответ:""".strip()


def _chunks_block(chunks: list[str]) -> str:
    if not chunks:
        return "(в базе знаний по вопросу ничего не найдено)"
    return "\n\n".join(f"[Источник {i}]\n{t}" for i, t in enumerate(chunks, 1))


def _docs_block(docs: list[Document], notes: dict[int, str]) -> str:
    if not docs:
        return ""

    if len(docs) == 1:
        body = docs[0].markdown
        if 0 in notes:
            body = f"{notes[0]}\n\n{body}"
        return f"Приложенный документ:\n[Документ: {docs[0].filename}]\n{body}"

    parts = []
    for i, d in enumerate(docs):
        body = d.markdown
        if i in notes:
            body = f"{notes[i]}\n\n{body}"
        parts.append(f"[Документ {i + 1} из {len(docs)}: {d.filename}]\n{body}")

    return "Приложенный документ:\n" + "\n\n".join(parts)


def build_prompt(
    *,
    question: str,
    instructions: str | None,
    history: list[tuple[str, str]],
    retrieved: list[dict],
    documents: list[Document],
    counter: TokenCounter,
    num_ctx: int,
    reserve_output: int,
    safety: int = 128,
    history_min_tokens: int = 2048,
    overflow: str = "truncate",
) -> BuiltPrompt:
    """Собирает промпт, укладываясь в num_ctx с запасом на генерацию.

    Приоритет вытеснения: инструкции и вопрос неприкосновенны, приложенные
    документы + чанки ЕПоЗ получают основную долю бюджета (fair-share),
    история жертвуется первой.
    """
    fixed = counter.count(
        _skeleton(HEAD_RAG, instructions, "", "", "", question)) + JOIN_SLACK
    free = num_ctx - reserve_output - safety - fixed

    if free < 0:
        needed = fixed + reserve_output + safety
        raise ContextOverflow(
            "Вопрос вместе с инструкциями не помещается в контекст модели: "
            f"нужно {needed} токенов при лимите {num_ctx}. Сократите вопрос "
            "или увеличьте CONTEXT_WINDOW.",
            needed=needed, available=num_ctx,
        )

    doc_costs = [
        (d.tokens if d.tokens is not None else counter.count(d.markdown))
        + counter.count(d.filename) + 12
        for d in documents
    ]
    chunk_costs = [
        counter.count(c.get("text", ""))
        + counter.count(c.get("source", "unknown")) + 12
        for c in retrieved
    ]

    costs = doc_costs + chunk_costs
    history_floor = min(history_min_tokens, free) if history else 0
    context_budget = max(0, free - history_floor)
    shares = _fit_shares(costs, context_budget)

    note_cost = counter.count(TRUNCATION_NOTE.format(percent=100)) + 4

    n_docs = len(documents)
    fitted_docs: list[Document] = []
    doc_notes: dict[int, str] = {}
    truncated_files: list[str] = []

    for i, (d, cost, share) in enumerate(
        zip(documents, doc_costs, shares[:n_docs])
    ):
        if share >= cost:
            fitted_docs.append(d)
            continue

        if overflow == "strict":
            raise ContextOverflow(
                f"Документ «{d.filename}» не помещается в контекст модели: "
                f"нужно {cost} токенов, доступно {share}.",
                needed=cost, available=share,
            )

        body = _truncate(d.markdown, max(0, share - note_cost), counter)
        percent = max(1, round(100 * len(body) / max(1, len(d.markdown))))
        doc_notes[i] = TRUNCATION_NOTE.format(percent=percent)
        fitted_docs.append(Document(d.filename, body))
        truncated_files.append(d.filename)

    # Чанки ЕПоЗ: поместились целиком — берём, иначе отбрасываем хвост.
    fitted_chunks: list[str] = []
    for c, share in zip(retrieved, shares[n_docs:]):
        text = c.get("text", "")
        cost = counter.count(text) + counter.count(c.get("source", "unknown")) + 12
        if share >= cost:
            fitted_chunks.append(text)

    context_block = _chunks_block(fitted_chunks)
    attached_block = _docs_block(fitted_docs, doc_notes)

    doc_used = counter.count(context_block) + counter.count(attached_block)

    history_budget = max(0, free - doc_used)
    kept: list[tuple[str, str]] = []
    used = 0

    for role, text in reversed(history):
        line = f"{'Пользователь' if role == 'user' else 'Ассистент'}: {text}"
        cost = counter.count(line) + 1
        if used + cost > history_budget:
            break
        used += cost
        kept.append((role, text))

    kept.reverse()
    while kept and kept[0][0] == "assistant":
        kept.pop(0)

    prompt = _skeleton(HEAD_RAG, instructions, context_block, attached_block,
                       _history_block(kept), question)

    sources = None
    if documents:
        sources = [
            d.filename + (PARTIAL_SUFFIX if d.filename in truncated_files else "")
            for d in documents
        ]

    stats = PromptStats(
        num_ctx=num_ctx,
        prompt_tokens=counter.count(prompt),
        reserved_output=reserve_output,
        documents=len(documents),
        truncated=len(truncated_files),
        document_tokens=doc_used,
        history_kept=len(kept),
        history_total=len(history),
        history_budget=history_budget,
        counter=getattr(counter, "name", "unknown"),
    )

    return BuiltPrompt(prompt, sources, stats, truncated_files)


HEAD_GENERATION = """Ты русскоязычный AI ассистент.
Отвечай ТОЛЬКО на русском языке.
Ты заполняешь шаблон документа на основании Единого положения о закупках (ЕПоЗ).
Заполняй только теми фактами, которые есть в приведённом контексте ЕПоЗ.
Чего в контексте нет — оставь как есть или явно пометь, что данные отсутствуют; НЕ выдумывай нормы, сроки и реквизиты.
Верни ПОЛНЫЙ заполненный документ в формате markdown, без пояснений и комментариев."""


def _generation_skeleton(head, instr, context_block, template_markdown) -> str:
    return f"""{head}
{instr}
Контекст ЕПоЗ:
{context_block}

Шаблон документа (заполни поля, верни полный документ):
{template_markdown}

Заполненный документ (markdown):""".strip()


def build_generation_prompt(
    *,
    template_markdown: str,
    instructions: str | None,
    retrieved: list[dict],
    counter: TokenCounter,
    num_ctx: int,
    reserve_output: int,
    safety: int = 128,
) -> str:
    """Промпт для генерации документа по шаблону: шаблон держим целиком (это
    структура результата), чанки ЕПоЗ подбираем в остаток окна."""
    instr = ""
    if instructions and instructions.strip():
        instr = f"Инструкция: {instructions.strip()}\n"

    fixed = counter.count(
        _generation_skeleton(HEAD_GENERATION, instr, "", template_markdown)
    ) + JOIN_SLACK
    free = num_ctx - reserve_output - safety - fixed

    if free < 0:
        needed = fixed + reserve_output + safety
        raise ContextOverflow(
            "Шаблон вместе с инструкциями не помещается в контекст модели: "
            f"нужно {needed} токенов при лимите {num_ctx}. Сократите шаблон "
            "или увеличьте CONTEXT_WINDOW.",
            needed=needed, available=num_ctx,
        )

    fitted: list[str] = []
    used = 0
    for c in retrieved:
        text = c.get("text", "")
        cost = counter.count(text) + counter.count(c.get("source", "unknown")) + 12
        if used + cost > free:
            continue
        used += cost
        fitted.append(text)

    context_block = _chunks_block(fitted)
    return _generation_skeleton(HEAD_GENERATION, instr, context_block, template_markdown)
