from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from contextlib import asynccontextmanager

from app.db.database import create_tables  
from app.api.chat import router as chat_router
from app.api.feedback import router as feedback_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # схему ведёт Alembic: alembic upgrade head перед стартом
    yield


app = FastAPI(
    title="RAG EPoZ Assistant API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router, tags=["chat"])
app.include_router(feedback_router, tags=["feedback"])


try:
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")
except RuntimeError:
    pass


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("dist/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        reload=True,
        timeout_keep_alive=300,
    )
