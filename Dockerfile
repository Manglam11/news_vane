# I build in two stages. The first stage installs dependencies with uv; the second
# copies only the finished environment across, so the shipped image carries no
# build tooling and no cache.

FROM python:3.12-slim AS builder

# uv ships as a standalone binary. I copy it from its official image rather than
# pip-installing it, which keeps this layer small and version-pinned.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# I copy the lockfile alone first. Docker caches this layer, so dependencies are
# only reinstalled when the lockfile actually changes -- not on every code edit.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Now the source. This layer rebuilds on every code change, which is cheap.
COPY src/ ./src/
COPY config/ ./config/
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

WORKDIR /app

# Running as root inside a container is a habit that gets punished in production.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder --chown=appuser:appuser /app /app

# The virtual environment's binaries go on PATH, so I can call uvicorn directly.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "newsvane.api.main:app", "--host", "0.0.0.0", "--port", "8000"]