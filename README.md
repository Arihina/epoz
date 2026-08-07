## Подготовка перед запуском
```
openssl req -x509 -newkey rsa:4096 \
-keyout key.pem \
-out cert.pem \
-days 365 \
-nodes
```
```
curl -fsSl https://ollama.com/install.sh | sh
```
```
ollama pull gemma2:9b
```
или
```
ollama pull qwen3.6:35b
```
```
sudo apt install pandoc
```
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
```
python3 create_collection_hybrid.py \
    --input_dir ./EPoZ_data \
    --chroma_dir ./chroma_db \
    --collection rag_docs_hybrid \
    --reset
```
```
sudo docker compose up -d
```
```
alembic upgrade head
```
```
python3 main.py
```

> Схему ведёт только Alembic — приложение не создаёт таблицы при старте. `alembic upgrade head` обязателен перед первым запуском, иначе сервис упадёт при первом запросе.

## Аутентификация

Сервис не управляет пользователями — это задача платформы (мастер-агент + Keycloak). RAG-ассистент получает UUID пользователя в заголовке `X-User-Id` и использует его как скоуп для своих данных. Заголовок обязателен **во всех** запросах:

```
X-User-Id: 11111111-1111-1111-1111-111111111111
```

JWT валидирует мастер-агент; RAG доверяет внутреннему трафику (сервис должен быть закрыт снаружи в обход мастера). При переходе на валидацию JWT по JWKS Keycloak меняется только `get_user_id`, эндпоинты не затрагиваются.

| Ситуация | Код |
|----------|-----|
| Заголовок `X-User-Id` отсутствует | `401` |
| `X-User-Id` не является валидным UUID | `401` |
| Обращение к чужому completion'у/response'у, чату (`conversation`) или фидбэку | `404` |

Возврат `404` (а не `403`) для чужих объектов сознателен: сервис не подтверждает их существование.

## API — общая идея

Сервис отдаёт генерацию в **двух** формах OpenAI API одновременно, плюс платформенное расширение для UI:

- **`/v1/chat/completions`** — форма Chat Completions. **Полностью stateless**: клиент присылает всю историю в `messages[]` при каждом запросе, сервис её не хранит и не переиспользует. Это путь для быстрой интеграции сторонних клиентов/вендоров и для отладки — им не нужно знать ничего специфичного про платформу.
- **`/v1/responses`** — форма Responses API. Тоже полностью рабочая сама по себе (можно слать весь `input` целиком, как `messages[]` у Chat Completions) — но именно эту форму использует собственный фронт платформы, потому что при передаче `conversation_id` **сервис сам собирает историю из БД**: клиент присылает только новый ход, не всю историю. Подробности — в разделе `POST /v1/responses` ниже.
- **`/v1/platform/conversations`** — платформенное (не входящее ни в одну спеку OpenAI) расширение для UI: список чатов, история сообщений, переименование, удаление. Не участвует в генерации напрямую — обе формы читают из тех же `conversations`/`chat_messages`, что здесь отображаются.

Фидбэк (`/v1/chat/completions/{id}/feedback`) и источники (`/v1/chat/completions/{id}/sources`) — **общие для обеих форм генерации**, не дублируются под `/v1/responses/...`: сообщение, сгенерированное через `/v1/responses`, получает `id` вида `resp_<uuid>`, но фидбэк/источники к нему адресуются по тому же общему пути, что и `chatcmpl-`-сообщения — сервис принимает оба префикса одинаково.

Сообщения персистятся в БД в любом случае (даже без `conversation_id`, в любой из форм) — это нужно для фидбэка/источников и повторного чтения, которые адресуются по `id` конкретного сообщения, а не по чату.

## База данных

PostgreSQL 16. Подключение настраивается через переменные окружения в `.env`:

```env
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=rag_db
```

Миграции управляются через Alembic. Применить:
```bash
alembic upgrade head
```

```
conversations                    платформенная сущность, только для UI/истории
├── id          UUID          PK, генерируется приложением (uuid4)
├── user_id     UUID          NOT NULL, индексирован — владелец чата
├── title       VARCHAR(255)  nullable (подставляется из первого вопроса, если не указан)
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

chat_messages
├── id                 UUID          PK, генерируется приложением (uuid4);
│                                    у ассистентских сообщений совпадает с
│                                    частью после "chatcmpl-"/"resp_" в id ответа
├── user_id            UUID          NOT NULL, индексирован — владелец записи
├── conversation_id    UUID          nullable, FK → conversations.id (ON DELETE CASCADE).
│                                    В форме Chat Completions — только привязка к чату
│                                    в UI-списке. В форме Responses — ЕЩЁ И источник
│                                    истории для генерации (см. POST /v1/responses)
├── role                VARCHAR(16)   "user" | "assistant"
├── content             TEXT
├── sources             JSONB         источники, реально использованные в ответе
├── retrieved_chunks    JSONB         весь пул извлечённых фрагментов до фильтрации (для /sources)
├── model               VARCHAR(64)   значение "model" из запроса
└── created_at          TIMESTAMPTZ

message_feedback
├── id          SERIAL        PK (идентификатор самой записи фидбэка, не используется в API)
├── message_id  UUID          FK → chat_messages.id (уникальный — одна оценка на сообщение)
├── vote        INTEGER       1 = лайк / -1 = дизлайк / NULL = без оценки
├── comment     TEXT          nullable
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ
```

Все идентификаторы — UUID, генерируются на стороне приложения (`default=uuid4`). `chat_messages` скоупится напрямую по `user_id`; принадлежность чату (`conversation_id`) опциональна. Удаление `conversation` каскадно удаляет её сообщения; удаление сообщения каскадно удаляет его фидбэк.

## Models
Так же модели настраиваются через `.env`
```
EMBED_MODEL=BAAI/bge-m3
OLLAMA_MODEL=qwen3.6:35b
HISTORY_LIMIT=10
```
или
```
EMBED_MODEL=intfloat/multilingual-e5-small
OLLAMA_MODEL=gemma2:9b
HISTORY_LIMIT=10
```
`HISTORY_LIMIT` — сколько последних сообщений чата подтягивать из БД в форме Responses при переданном `conversation_id` (см. ниже). На форму Chat Completions не влияет — там истории из БД не бывает вообще, только из тела запроса.

## API

Все эндпоинты требуют заголовок `X-User-Id: <uuid>`. Ресурсы скоупятся по этому идентификатору; обращение к чужому ресурсу возвращает `404`.

### `POST /v1/chat/completions`

Генерация ответа, форма Chat Completions. Клиент присылает **всю** историю диалога в `messages[]` — сервис её не хранит и не переиспользует между запросами, независимо от того, передан ли `conversation_id`.

Тело запроса:
```json
{
  "model": "epoz",
  "messages": [
    {"role": "user", "content": "что такое РАГ"},
    {"role": "assistant", "content": "РАГ — это..."},
    {"role": "user", "content": "а какие сроки"}
  ],
  "stream": true,
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```
| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `model` | string | нет (по умолчанию `"epoz"`) | не влияет на поведение, только эхом в ответе |
| `messages` | array | да | последнее сообщение — `role: "user"`, это и есть текущий вопрос |
| `stream` | bool | нет (по умолчанию `false`) | стримить ответ через SSE |
| `conversation_id` | UUID string | нет | платформенное расширение — привязать сообщение к чату из `/v1/platform/conversations`. **Только ярлык для UI** — контекст для генерации в этой форме всегда из `messages[]`, из БД ничего не подтягивается (в отличие от формы Responses ниже). Чужой/несуществующий `conversation_id` → `404` |

**Нестрим-ответ** (`chat.completion`):
```json
{
  "id": "chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "object": "chat.completion",
  "created": 1735900000,
  "model": "epoz",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "РАГ (Retrieval-Augmented Generation) — это...\n\nИсточники:\n- doc1.pdf"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 412, "completion_tokens": 87, "total_tokens": 499}
}
```

**Стрим-ответ** (`stream: true`) — SSE, `chat.completion.chunk`, один и тот же `id` во всех чанках:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1735900000,"model":"epoz","conversation_id":"3fa85f64-...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1735900000,"model":"epoz","conversation_id":"3fa85f64-...","choices":[{"index":0,"delta":{"content":"РАГ"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1735900000,"model":"epoz","conversation_id":"3fa85f64-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```
`conversation_id` в ответе — `null`, если не был передан в запросе. `id` (`chatcmpl-<uuid>`) — ключ для фидбэка и источников ниже. Блок «Источники: ...» добавляется в `content` в конце ответа, если ассистент использовал извлечённые фрагменты (как обычный текст, не отдельным полем).

---

### `GET /v1/chat/completions/{completion_id}`

Получить ранее сгенерированный ответ повторно (в форме `chat.completion`) — по `id`, который пришёл в ответе. Работает для сообщений, сгенерированных **любой** из двух форм: `chatcmpl-<uuid>` целиком, `resp_<uuid>` целиком, или голый UUID.

Ответ — тот же `chat.completion`-объект, что и у `POST` в нестрим-режиме, восстановленный из БД (включая `usage`):
```json
{
  "id": "chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "object": "chat.completion",
  "created": 1735900000,
  "model": "epoz",
  "conversation_id": null,
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "РАГ (Retrieval-Augmented Generation) — это...\n\nИсточники:\n- doc1.pdf"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 412, "completion_tokens": 87, "total_tokens": 499}
}
```
Если `completion_id` указывает на сообщение с `role: "user"` (в норме такой `id` клиенту никогда не отдаётся) — `404`, как и для несуществующего/чужого id.

---

### `POST /v1/responses`

Генерация ответа, форма Responses API. Есть два режима работы в зависимости от того, передан ли `conversation_id`:

- **Без `conversation_id`** — честный stateless-режим: `input` должен содержать всё, что нужно модели (аналог `messages[]`), сервис ничего не подтягивает из БД. Подходит для тех же сценариев, что и `/v1/chat/completions` — быстрая интеграция, отладка.
- **С `conversation_id`** — `input` должен содержать **только новый ход** (одну реплику пользователя, без истории). Сервис сам вызывает `get_recent_conversation_messages`, читает последние `HISTORY_LIMIT` сообщений этого чата из БД и подставляет их как историю. Если клиент всё равно пришлёт историю в `input` вместе с `conversation_id` — `422`, а не молчаливый выбор одной из версий:
  ```json
  {"error": {"message": "При переданном conversation_id input должен содержать только новый ход (без истории) — история собирается агентом из БД по conversation_id", ...}}
  ```

Тело запроса (`input` строкой — простой вопрос без истории):
```json
{
  "model": "epoz",
  "input": "что такое РАГ",
  "stream": true,
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```
Тело запроса (`input` списком items — без `conversation_id`, история целиком от клиента):
```json
{
  "model": "epoz",
  "input": [
    {"role": "user", "content": "что такое РАГ"},
    {"role": "assistant", "content": [{"type": "output_text", "text": "РАГ — это..."}]},
    {"role": "user", "content": "а какие сроки"}
  ]
}
```
Части `content`-массива: `"input_text"` — новый ввод клиента, `"output_text"` — то, чем клиент эхом возвращает прошлый ответ модели при ручном ведении истории (стандартный способ у Responses API). `instructions` в теле принимается, но осознанно не используется — у epoz фиксированный промпт под свою предметную область, как и было в форме Chat Completions.

**Нестрим-ответ**:
```json
{
  "id": "resp_1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "object": "response",
  "created_at": 1735900000,
  "status": "completed",
  "model": "epoz",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "output": [{
    "id": "msg_1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "РАГ — это...\n\nИсточники:\n- doc1.pdf", "annotations": []}]
  }],
  "usage": {"input_tokens": 412, "output_tokens": 87, "total_tokens": 499}
}
```
Обратите внимание: `usage` здесь — `input_tokens`/`output_tokens` (терминология Responses API), а не `prompt_tokens`/`completion_tokens`, как у Chat Completions — те же числа, разные имена полей, это не расхождение, а разница спек.

**Стрим-ответ** (`stream: true`) — гранулярные типизированные SSE-события (не единый тип чанка, как у Chat Completions):
```
event: response.created
data: {"type":"response.created","sequence_number":1,"response":{"id":"resp_...","object":"response","status":"in_progress",...}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{...}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"item_id":"msg_...","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_...","output_index":0,"content_index":0,"delta":"РАГ"}

... (ещё response.output_text.delta на каждый токен) ...

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":N,"item_id":"msg_...","output_index":0,"content_index":0,"text":"РАГ — это...\n\nИсточники:\n- doc1.pdf"}

event: response.content_part.done
data: {...}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":N,"output_index":0,"item":{...,"status":"completed"}}

event: response.completed
data: {"type":"response.completed","sequence_number":N,"response":{"id":"resp_...","status":"completed","output":[...],"usage":{...}}}
```
`sequence_number` — монотонно растущий счётчик событий потока (сквозной для всего ответа, не по типам). Обрыв генерации / внутренняя ошибка — отдельное `event: error` с `sequence_number: 9999` (зарезервирован специально под ошибку, чтобы не путать со штатной нумерацией).

---

### `GET /v1/responses/{completion_id}`

То же самое, что `GET /v1/chat/completions/{completion_id}`, но возвращает `response`-объект (форма Responses), а не `chat.completion`. Принимает тот же id в любом виде (`resp_<uuid>`, `chatcmpl-<uuid>`, голый UUID) — оба GET-эндпоинта читают одну и ту же таблицу, просто по-разному сериализуют.

---

### `POST/GET/DELETE /v1/chat/completions/{completion_id}/feedback`

Оценка ответа ассистента — **общий путь для обеих форм генерации**, не дублируется под `/v1/responses/...`. `{completion_id}` принимает `chatcmpl-<uuid>`, `resp_<uuid>` или голый UUID одинаково.

Тело `POST`-запроса (все поля опциональны, повторный вызов обновляет существующую оценку):
```json
{ "vote": 1, "comment": "Очень подробно" }
```
| Поле | Тип | Значения |
|---|---|---|
| `vote` | integer | `1` — лайк, `-1` — дизлайк, `null`/отсутствует — без оценки |
| `comment` | string | любой текст, опционально |

Ответ (`POST`/`GET`):
```json
{
  "message_id": "1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "vote": 1,
  "comment": "Очень подробно",
  "created_at": "2026-08-03T12:01:00",
  "updated_at": "2026-08-03T12:01:00"
}
```
`DELETE` сбрасывает оценку (`vote=null`, `comment=null`) и возвращает `204`.

---

### `GET /v1/chat/completions/{completion_id}/sources`

Источники, использованные в конкретном ответе — тоже общий путь для обеих форм.
```json
{
  "id": "chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "retrieved": [{"text": "...", "source": "doc1.pdf", "score": 0.87}],
  "used_sources": ["doc1.pdf"]
}
```
`retrieved` — весь пул извлечённых фрагментов до фильтрации; `used_sources` — то, что действительно попало в ответ (та же эвристика — пересечение слов ответа с текстом фрагмента).

---

### Чаты — `/v1/platform/conversations`

Платформенное расширение для UI (список чатов, история, переименование, удаление). **Не входит ни в одну спеку OpenAI** и не участвует в генерации напрямую — см. «API — общая идея» выше. Общее для обеих форм: сообщения, привязанные к чату через `conversation_id` в любой из форм, попадают в одну и ту же историю.

#### `POST /v1/platform/conversations`
Создать новый чат.

Тело запроса (опционально): `{ "title": "Название чата" }`

Ответ:
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "title": "Название чата",
  "created_at": "2026-08-03T12:00:00",
  "updated_at": "2026-08-03T12:00:00"
}
```

#### `GET /v1/platform/conversations`
Список чатов текущего пользователя, отсортированных по дате последнего сообщения (новые первые). Формат элемента — как у `POST`.

#### `GET /v1/platform/conversations/{id}/messages`
История сообщений чата вместе с фидбэком — для восстановления `messages[]`/`input` на фронте при открытии чата (или, в форме Responses с `conversation_id`, просто для отображения — сервис сам подтянет то же самое для генерации).
```json
[
  {
    "id": "9c858901-8a57-4791-81fe-4c455b099bc9",
    "role": "user",
    "content": "Что такое РАГ",
    "sources": [],
    "created_at": "2026-08-03T12:00:00",
    "feedback": null
  },
  {
    "id": "1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
    "role": "assistant",
    "content": "РАГ (Retrieval-Augmented Generation) — это...",
    "sources": ["doc1.pdf"],
    "created_at": "2026-08-03T12:00:05",
    "feedback": {"vote": 1, "comment": "Хороший ответ"}
  }
]
```
`id` ассистентского сообщения используется для `/v1/chat/completions/{id}/feedback`/`/sources` и для `GET`-по-id в любой из форм.

#### `PATCH /v1/platform/conversations/{id}`
Переименовать чат. Тело: `{ "title": "Новое название" }`.

#### `DELETE /v1/platform/conversations/{id}`
Удалить чат со всеми сообщениями и их фидбэком (каскадно). Ответ: `204 No Content`.

---

## Ошибки

Единый формат вместо FastAPI-дефолта `{"detail": ...}`:
```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```
`type` — грубая классификация по HTTP-статусу: `400/413/415/422` → `invalid_request_error`, `401` → `authentication_error`, `404` → `not_found_error`, остальное → `server_error`.

---

## Примеры curl

```bash
U=11111111-1111-1111-1111-111111111111

# ============ форма Chat Completions ============

# генерация, без привязки к чату
curl -k -X POST https://localhost:8443/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "messages": [{"role": "user", "content": "что такое Меры ограничительного характера"}]}'
# -> {"id": "chatcmpl-...", "object": "chat.completion", "choices": [...], "usage": {...}}

# то же самое, стримом
curl -k -N -X POST https://localhost:8443/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "stream": true, "messages": [{"role": "user", "content": "что такое РАГ"}]}'

# продолжение диалога — история целиком в теле
curl -k -X POST https://localhost:8443/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "messages": [
        {"role": "user", "content": "что такое РАГ"},
        {"role": "assistant", "content": "РАГ (Retrieval-Augmented Generation) — это..."},
        {"role": "user", "content": "а какие сроки"}
      ]}'

# ============ форма Responses ============

# генерация без conversation_id — input строкой, аналог messages[] из одного вопроса
curl -k -N -X POST https://localhost:8443/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "stream": true, "input": "что такое РАГ"}'

# создать чат, затем спросить в нём — история дальше ведётся сервером
curl -k -X POST https://localhost:8443/v1/platform/conversations \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Тестовый чат"}'
# -> {"id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", ...}

CID=3fa85f64-5717-4562-b3fc-2c963f66afa6

curl -k -X POST https://localhost:8443/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"epoz\", \"conversation_id\": \"$CID\", \"input\": \"что такое РАГ\"}"

# следующий вопрос в том же чате — снова только новый ход, БЕЗ истории
curl -k -X POST https://localhost:8443/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"epoz\", \"conversation_id\": \"$CID\", \"input\": \"а какие сроки\"}"

ID=resp_1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2

# получить ответ повторно (форма Responses)
curl -k https://localhost:8443/v1/responses/$ID -H "X-User-Id: $U"

# ============ общее для обеих форм ============

# получить ответ повторно как chat.completion, независимо от того, какой формой сгенерирован
curl -k https://localhost:8443/v1/chat/completions/$ID -H "X-User-Id: $U"

# источники
curl -k https://localhost:8443/v1/chat/completions/$ID/sources -H "X-User-Id: $U"

# фидбэк
curl -k -X POST https://localhost:8443/v1/chat/completions/$ID/feedback \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"vote": 1, "comment": "Хороший ответ"}'
curl -k https://localhost:8443/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"
curl -k -X DELETE https://localhost:8443/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"

# чаты
curl -k https://localhost:8443/v1/platform/conversations -H "X-User-Id: $U"
curl -k https://localhost:8443/v1/platform/conversations/$CID/messages -H "X-User-Id: $U"
curl -k -X PATCH https://localhost:8443/v1/platform/conversations/$CID \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Новое название"}'
curl -k -X DELETE https://localhost:8443/v1/platform/conversations/$CID -H "X-User-Id: $U"

# ============ ошибки ============

# без X-User-Id -> 401 в едином формате
curl -k -X POST https://localhost:8443/v1/chat/completions \
  -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "привет"}]}'
# -> {"error": {"message": "...", "type": "authentication_error", "param": null, "code": null}}

# conversation_id + история в input одновременно (форма Responses) -> 422
curl -k -X POST https://localhost:8443/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"epoz\", \"conversation_id\": \"$CID\", \"input\": [
        {\"role\": \"user\", \"content\": \"вопрос 1\"},
        {\"role\": \"assistant\", \"content\": \"ответ 1\"},
        {\"role\": \"user\", \"content\": \"вопрос 2\"}
      ]}"
# -> {"error": {"message": "При переданном conversation_id input должен содержать только новый ход...", "type": "invalid_request_error", "param": null, "code": null}}

# чужой/несуществующий completion_id -> 404
curl -k https://localhost:8443/v1/chat/completions/chatcmpl-00000000-0000-0000-0000-000000000000/sources \
  -H "X-User-Id: $U"
# -> {"error": {"message": "Сообщение не найдено", "type": "not_found_error", "param": null, "code": null}}

# пустой messages -> 422
curl -k -X POST https://localhost:8443/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"messages": []}'
# -> {"error": {"message": "messages обязателен и не должен быть пустым", "type": "invalid_request_error", "param": null, "code": null}}
```