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
python3 create_collection.py
```
```
python3 main.py
```

## База данных
 
Файл `rag_history.db` создаётся автоматически при первом запуске рядом с `main.py`.
 
```
chat_sessions
├── id          INTEGER  PK autoincrement
├── title       TEXT     nullable (подставляется из первого вопроса если не указан)
├── created_at  DATETIME
└── updated_at  DATETIME
 
chat_messages
├── id          INTEGER  PK autoincrement
├── session_id  INTEGER  FK → chat_sessions.id
├── role        TEXT     "user" | "assistant"
├── content     TEXT
├── sources     TEXT     JSON-список источников, например ["doc1.pdf"] или NULL
└── created_at  DATETIME
 
message_feedback
├── id          INTEGER  PK autoincrement
├── message_id  INTEGER  FK → chat_messages.id (уникальный — одна оценка на сообщение)
├── vote        INTEGER  1 = лайк / -1 = дизлайк / NULL = без оценки
├── comment     TEXT     nullable
├── created_at  DATETIME
└── updated_at  DATETIME
```
 
Удаление каскадное

## API
 
### Чаты
 
#### `POST /sessions`
Создать новый чат.
 
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
Список всех чатов, отсортированных по дате последнего сообщения (новые первые).
 
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
data: {"token": "RAG"}
data: {"token": " — это"}
data: {"token": " метод..."}
...
data: {"token": "\n\nИсточники:\n- doc1.pdf"}
data: {"message_id": 2}
data: [DONE]
```
 
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
 
```bash
# Создать чат
curl -k -X POST https://localhost:8443/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Тестовый чат"}'
 
# Список чатов
curl -k https://localhost:8443/sessions
 
# Отправить вопрос
curl -k -X POST https://localhost:8443/sessions/1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Что такое RAG"}' \
  --no-buffer
 
# История сообщений
curl -k https://localhost:8443/sessions/1/messages
 
# Лайк с комментарием
curl -k -X POST https://localhost:8443/messages/1/feedback \
  -H "Content-Type: application/json" \
  -d '{"vote": 1, "comment": "Хороший ответ"}'
 
# Только комментарий
curl -k -X POST https://localhost:8443/messages/1/feedback \
  -H "Content-Type: application/json" \
  -d '{"comment": "Ответ неточный"}'
 
# Получить фидбэк
curl -k https://localhost:8443/messages/1/feedback
 
# Переименовать чат
curl -k -X PATCH https://localhost:8443/sessions/1 \
  -H "Content-Type: application/json" \
  -d '{"title": "Новое название"}'
 
# Удалить чат
curl -k -X DELETE https://localhost:8443/sessions/1
```
 