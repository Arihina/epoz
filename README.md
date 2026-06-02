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
docker compose -d
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
| Обращение к чужому чату/сообщению | `404` |

Возврат `404` (а не `403`) для чужих объектов сознателен: сервис не подтверждает их существование.

## База данных

PostgreSQL 16. Подключение настраивается через переменные окружения в `.env`:

```env
DB_HOST=db
DB_PORT=5432
DB_USER=rag_user
DB_PASSWORD=rag_pass
DB_NAME=rag_db
```

Миграции управляются через Alembic. Применить:
```bash
alembic upgrade head
```

```
chat_sessions
├── id          SERIAL        PK
├── user_id     UUID          NOT NULL, индексирован — владелец чата
├── title       VARCHAR(255)  nullable (подставляется из первого вопроса если не указан)
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

chat_messages
├── id          SERIAL        PK
├── session_id  INTEGER       FK → chat_sessions.id
├── role        VARCHAR(16)   "user" | "assistant"
├── content     TEXT
├── sources     JSONB         список источников
└── created_at  TIMESTAMPTZ

message_feedback
├── id          SERIAL        PK
├── message_id  INTEGER       FK → chat_messages.id (уникальный — одна оценка на сообщение)
├── vote        INTEGER       1 = лайк / -1 = дизлайк / NULL = без оценки
├── comment     TEXT          nullable
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ
```

Принадлежность пользователю хранится только в `chat_sessions.user_id`. Сообщения и фидбэк скоупятся транзитивно через FK на сессию. Удаление каскадное. `sources` хранится как нативный JSONB — десериализация на стороне приложения не требуется.

## API

Все эндпоинты требуют заголовок `X-User-Id: <uuid>`. Списки и операции скоупятся по этому идентификатору; обращение к чужому ресурсу возвращает `404`.

### Чаты
 
#### `POST /sessions`
Создать новый чат (привязывается к `X-User-Id`).
 
Тело запроса (опционально):
```json
{ "title": "Название чата" }
```
 
Ответ:
```json
{
  "id": 1,
  "title": "Название чата",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:00"
}
```
 
---
 
#### `GET /sessions`
Список чатов текущего пользователя, отсортированных по дате последнего сообщения (новые первые).
 
Ответ:
```json
[
  {
    "id": 1,
    "title": "Что такое RAG",
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:05:00"
  }
]
```
 
---
 
#### `GET /sessions/{session_id}/messages`
История сообщений чата вместе с фидбэком.
 
Ответ:
```json
[
  {
    "id": 1,
    "role": "user",
    "content": "Что такое RAG",
    "sources": [],
    "created_at": "2024-01-01T12:00:00",
    "feedback": null
  },
  {
    "id": 2,
    "role": "assistant",
    "content": "RAG (Retrieval-Augmented Generation) — это...",
    "sources": ["doc1.pdf"],
    "created_at": "2024-01-01T12:00:05",
    "feedback": {
      "vote": 1,
      "comment": "Хороший ответ"
    }
  }
]
```
 
#### `PATCH /sessions/{session_id}`
Переименовать чат.
 
Тело запроса:
```json
{ "title": "Новое название" }
```
 
Ответ:
```json
{
  "id": 1,
  "title": "Новое название",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:10:00"
}
```

---
 
#### `DELETE /sessions/{session_id}`
Удалить чат со всеми сообщениями и фидбэком.
 
Ответ: `204 No Content`
 
---
 
### Сообщения
 
#### `POST /sessions/{session_id}/chat`
Отправить вопрос. Ответ возвращается потоком (SSE).

Тело запроса:
```json
{ "message": "Что такое RAG?" }
```

Поток событий:
```
data: {"chunks": [{"text": "RAG (Retrieval-Augmented Generation) — метод...", "source": "doc1.pdf", "score": 0.87}, ...]}
data: {"token": "RAG"}
data: {"token": " — это"}
data: {"token": " метод..."}
...
data: {"token": "\n\nИсточники:\n- doc1.pdf"}
data: {"message_id": 2}
data: [DONE]
```

Порядок событий:
- `chunks` — **первое** событие, приходит до начала генерации; содержит список извлечённых фрагментов, которые использовались как контекст для ответа (только при RAG-запросе, при small-talk отсутствует)
- `token` — токены ответа LLM, включая блок источников в конце
- `message_id` — ID сохранённого сообщения, используется для отправки фидбэка
- `[DONE]` — завершение потока
 
> `message_id` из предпоследнего события используется для отправки фидбэка.
 
---
 
### Фидбэк
 
#### `POST /messages/{message_id}/feedback`
Поставить оценку и/или комментарий к ответу ассистента. Все поля опциональны. Повторный вызов обновляет существующую оценку.
 
Тело запроса:
```json
{ "vote": 1, "comment": "Очень подробно" }
```
 
| Поле      | Тип     | Значения                              |
|-----------|---------|---------------------------------------|
| `vote`    | integer | `1` — лайк, `-1` — дизлайк, `null` или отсутствует — без оценки |
| `comment` | string  | любой текст, опционально              |
 
Ответ:
```json
{
  "message_id": 2,
  "vote": 1,
  "comment": "Очень подробно",
  "created_at": "2024-01-01T12:01:00",
  "updated_at": "2024-01-01T12:01:00"
}
```
 
---
 
#### `GET /messages/{message_id}/feedback`
Получить текущую оценку сообщения.
 
---
 
#### `DELETE /messages/{message_id}/feedback`
Сбросить оценку и комментарий (устанавливает `vote=null`, `comment=null`).
 
Ответ: `204 No Content`
 
---
 
## Примеры curl

Во всех запросах передаётся `X-User-Id`.
 
```bash
U=11111111-1111-1111-1111-111111111111

# Создать чат
curl -k -X POST https://localhost:8443/sessions \
  -H "X-User-Id: $U" \
  -H "Content-Type: application/json" \
  -d '{"title": "Тестовый чат"}'
 
# Список чатов
curl -k https://localhost:8443/sessions \
  -H "X-User-Id: $U"
 
# Отправить вопрос
curl -k -X POST https://localhost:8443/sessions/1/chat \
  -H "X-User-Id: $U" \
  -H "Content-Type: application/json" \
  -d '{"message": "Что такое Меры ограничительного характера"}' \
  --no-buffer
 
# История сообщений
curl -k https://localhost:8443/sessions/1/messages \
  -H "X-User-Id: $U"
 
# Лайк с комментарием
curl -k -X POST https://localhost:8443/messages/1/feedback \
  -H "X-User-Id: $U" \
  -H "Content-Type: application/json" \
  -d '{"vote": 1, "comment": "Хороший ответ"}'
 
# Только комментарий
curl -k -X POST https://localhost:8443/messages/1/feedback \
  -H "X-User-Id: $U" \
  -H "Content-Type: application/json" \
  -d '{"comment": "Ответ неточный"}'
 
# Получить фидбэк
curl -k https://localhost:8443/messages/1/feedback \
  -H "X-User-Id: $U"
 
# Переименовать чат
curl -k -X PATCH https://localhost:8443/sessions/1 \
  -H "X-User-Id: $U" \
  -H "Content-Type: application/json" \
  -d '{"title": "Новое название"}'
 
# Удалить чат
curl -k -X DELETE https://localhost:8443/sessions/1 \
  -H "X-User-Id: $U"
```
