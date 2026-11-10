# MinerU для epoz (локальная разработка, CPU-сборка).
#
# Порт Dockerfile из document_agent с одним отличием: дефолт torch — CPU
# (https://download.pytorch.org/whl/cpu), чтобы локально не тащить CUDA-колёса
# (~3 ГБ разницы). Для GPU-сборки пересобрать с
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#
# Версия закреплена жёстко: начиная с 4.0 у MinerU другой API (только /v1, без
# /file_parse) и другой CLI моделей (без -m/-s), а extras core/pipeline в 4.x
# вообще отсутствуют — pip на них лишь предупреждает и ставит 4.x, после чего
# агент получает 404 на каждый документ. 2.7.6 — последняя 2.x, с которой
# написан document_agent.
#
# Официальный образ MinerU собран на vllm/vllm-openai — оправдано для
# vlm/hybrid-бэкендов, но лишнее для "pipeline"-бэкенда (кириллица через явную
# recognition-модель): он тянет torch + torchvision + onnxruntime и НЕ тянет
# paddle. Поэтому базовый python-образ: CUDA-рантайм (если нужен) приезжает с
# колёсами torch, драйвер пробрасывается с хоста через nvidia-container-toolkit.

FROM python:3.12-slim-bookworm

# libgl1/libglib2.0 — стандартная нужда OpenCV/Paddle-зависимостей внутри
# mineru[pipeline], без них падает на "libGL.so.1: cannot open shared object
# file" при первом вызове разбора документа.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Вариант torch задаётся явно, а не отдаётся на откуп дефолту PyPI:
#   только CPU: https://download.pytorch.org/whl/cpu   (дефолт здесь, образ легче)
#   CUDA 12.4 : https://download.pytorch.org/whl/cu124
#   CUDA 12.8 : https://download.pytorch.org/whl/cu128
# Версия должна быть не новее CUDA-драйвера на хосте (`nvidia-smi`).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL}

# torch уже стоит нужного варианта — mineru его не переустановит.
RUN pip install --no-cache-dir "mineru[core,pipeline]==2.7.6"

# huggingface по умолчанию; если сервер не видит HF — пересобрать с
# --build-arg MINERU_MODEL_SOURCE=modelscope
ARG MINERU_MODEL_SOURCE=huggingface

# Модели тянутся один раз при сборке образа, а не при каждом старте контейнера.
# -m pipeline: только модели pipeline-бэкенда, без vlm.
RUN mineru-models-download -s ${MINERU_MODEL_SOURCE} -m pipeline

# После сборки образ самодостаточен — mineru.json уже указывает на локальные
# пути моделей, сеть при старте контейнера не нужна.
ENV MINERU_MODEL_SOURCE=local

EXPOSE 8010

# /health у MinerU 2.x нет — единственный маршрут POST /file_parse, поэтому
# проверяем /openapi.json (его FastAPI отдаёт всегда).
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8010/openapi.json || exit 1

CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8010"]
