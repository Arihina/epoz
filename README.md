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
| Обращение к чужому completion'у, чату (`conversation`) или фидбэку | `404` |

Возврат `404` (а не `403`) для чужих объектов сознателен: сервис не подтверждает их существование.

## API — общая идея

API состоит из двух независимых частей:

- **`/v1/chat/completions`** — OpenAI-совместимый эндпоинт генерации. **Полностью stateless**: сервис не хранит и не переиспользует историю диалога — клиент присылает её целиком в `messages[]` при каждом запросе. Формат запроса/ответа соответствует `chat.completion` / `chat.completion.chunk` из OpenAI Chat Completions API.
- **`/v1/platform/conversations`** — платформенное (не входящее в OpenAI-стандарт) расширение для UI: список чатов, история сообщений, переименование, удаление. Не участвует в генерации и не хранит контекст для модели — это только группировка сообщений для отображения. Связь между двумя частями — необязательное поле `conversation_id` в теле запроса к `/v1/chat/completions`.


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
│                                    частью после "chatcmpl-" в id completion'а
├── user_id            UUID          NOT NULL, индексирован — владелец записи
├── conversation_id    UUID          nullable, FK → conversations.id (ON DELETE CASCADE).
│                                    НЕ участвует в сборке контекста для генерации —
│                                    только привязка к чату в UI-списке
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

Все идентификаторы — UUID, генерируются на стороне приложения (`default=uuid4`). `chat_messages` скоупится напрямую по `user_id`; принадлежность чату (`conversation_id`) опциональна и не связана с тем, кто владеет записью. Удаление `conversation` каскадно удаляет её сообщения; удаление сообщения каскадно удаляет его фидбэк.

## Models
Так же модели настраиваются через `.env`
```
EMBED_MODEL=BAAI/bge-m3
OLLAMA_MODEL=qwen3.6:35b
```
или
```
EMBED_MODEL=intfloat/multilingual-e5-small
OLLAMA_MODEL=gemma2:9b
```

## API

Все эндпоинты требуют заголовок `X-User-Id: <uuid>`. Ресурсы скоупятся по этому идентификатору; обращение к чужому ресурсу возвращает `404`.

### `POST /v1/chat/completions`

Генерация ответа. Клиент присылает **всю** историю диалога в `messages[]` — сервис её не хранит и не переиспользует между запросами.

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
| `conversation_id` | UUID string | нет | платформенное расширение — привязать сообщение к чату из `/v1/platform/conversations`. Чужой/несуществующий `conversation_id` → `404` |

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

### `POST/GET/DELETE /v1/chat/completions/{completion_id}/feedback`

Оценка ответа ассистента. `{completion_id}` — значение `id` из ответа (`chatcmpl-<uuid>` целиком или голый UUID — оба варианта принимаются).

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

Источники, использованные в конкретном ответе — замена ушедшему из стрима событию `chunks`.
```json
{
  "id": "chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "retrieved": [{"text": "...", "source": "doc1.pdf", "score": 0.87}],
  "used_sources": ["doc1.pdf"]
}
```
`retrieved` — весь пул извлечённых фрагментов до фильтрации; `used_sources` — то, что действительно попало в ответ (та же эвристика, что и раньше — пересечение слов ответа с текстом фрагмента).

---

### Чаты — `/v1/platform/conversations`

Платформенное расширение для UI (список чатов, история, переименование, удаление). **Не входит в OpenAI-стандарт** и не участвует в генерации — см. «API — общая идея» выше.

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
История сообщений чата вместе с фидбэком — для восстановления `messages[]` на фронте при открытии чата.
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
`id` ассистентского сообщения — тот же UUID, что использовать для `/v1/chat/completions/{id}/feedback` и `/sources`.

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
 
# --- генерация, без привязки к чату ---
curl -k -X POST http://localhost:8001/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "messages": [{"role": "user", "content": "что такое Меры ограничительного характера"}]}'
# -> {"id": "chatcmpl-...", "object": "chat.completion", "choices": [...], "usage": {...}}
 
# то же самое, стримом
curl -k -N -X POST http://localhost:8001/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "stream": true, "messages": [{"role": "user", "content": "что такое РАГ"}]}'
 
# продолжение диалога — история целиком в теле
curl -k -X POST http://localhost:8001/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "epoz", "messages": [
        {"role": "user", "content": "что такое РАГ"},
        {"role": "assistant", "content": "РАГ (Retrieval-Augmented Generation) — это..."},
        {"role": "user", "content": "а какие сроки"}
      ]}'
 
ID=chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2
 
# получить ответ повторно по id (например, если клиент потерял тело исходного ответа)
curl -k http://localhost:8001/v1/chat/completions/$ID -H "X-User-Id: $U"
 
# источники
curl -k http://localhost:8001/v1/chat/completions/$ID/sources -H "X-User-Id: $U"
 
# --- фидбэк ---
 
# поставить оценку
curl -k -X POST http://localhost:8001/v1/chat/completions/$ID/feedback \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"vote": 1, "comment": "Хороший ответ"}'
 
# посмотреть текущую оценку
curl -k http://localhost:8001/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"
 
# сбросить оценку
curl -k -X DELETE http://localhost:8001/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"
 
# --- чаты (платформенный CRUD) ---
 
# создать чат
curl -k -X POST http://localhost:8001/v1/platform/conversations \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Тестовый чат"}'
# -> {"id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", ...}
 
CID=3fa85f64-5717-4562-b3fc-2c963f66afa6
 
# сообщение внутри чата — conversation_id привязывает запись к нему
curl -k -X POST http://localhost:8001/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"epoz\", \"conversation_id\": \"$CID\", \"messages\": [{\"role\": \"user\", \"content\": \"что такое РАГ\"}]}"
 
# список чатов
curl -k http://localhost:8001/v1/platform/conversations -H "X-User-Id: $U"
 
# история сообщений чата
curl -k http://localhost:8001/v1/platform/conversations/$CID/messages -H "X-User-Id: $U"
 
# переименовать / удалить
curl -k -X PATCH http://localhost:8001/v1/platform/conversations/$CID \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Новое название"}'
curl -k -X DELETE http://localhost:8001/v1/platform/conversations/$CID -H "X-User-Id: $U"
 
# --- ошибки ---
 
# без X-User-Id -> 401 в едином формате
curl -k -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "привет"}]}'
# -> {"error": {"message": "...", "type": "authentication_error", "param": null, "code": null}}
 
# чужой/несуществующий completion_id -> 404
curl -k http://localhost:8001/v1/chat/completions/chatcmpl-00000000-0000-0000-0000-000000000000/sources \
  -H "X-User-Id: $U"
# -> {"error": {"message": "Сообщение не найдено", "type": "not_found_error", "param": null, "code": null}}
 
# пустой messages -> 422
curl -k -X POST http://localhost:8001/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"messages": []}'
# -> {"error": {"message": "messages обязателен и не должен быть пустым", "type": "invalid_request_error", "param": null, "code": null}}
```