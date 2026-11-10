FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy RAG_MODELS_DIR=/app/models
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# Models are pinned by revision + sha256; the index is built here, offline, from the committed corpus.
RUN .venv/bin/rag-normas models
COPY data/corpus ./data/corpus
RUN .venv/bin/rag-normas build

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=build --chown=app:app /app/src /app/src
COPY --from=build --chown=app:app /app/models /app/models
COPY --from=build --chown=app:app /app/data/index-e5-small-int8.sqlite /app/data/index-e5-small-int8.sqlite
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8000 \
    RAG_MODELS_DIR=/app/models RAG_INDEX=/app/data/index-e5-small-int8.sqlite \
    RAG_ORT_THREADS=1 RAG_MAX_CONCURRENCY=1 MALLOC_ARENA_MAX=2 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
ENTRYPOINT ["rag-normas"]
CMD ["serve"]
