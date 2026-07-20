# I build in two stages. The first stage installs dependencies with uv and pulls the
# serving weights; the second copies only the finished environment across, so the
# shipped image carries no build tooling, no curl and no cache.

FROM python:3.12-slim AS builder

# uv ships as a standalone binary. I copy it from its official image rather than
# pip-installing it, which keeps this layer small and version-pinned.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# The slim base ships no curl, and the weights live outside the repo. This sits above
# the source copies on purpose so Docker caches it and does not reinstall on every edit.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# I copy the lockfile alone first. Docker caches this layer, so dependencies are
# only reinstalled when the lockfile actually changes -- not on every code edit.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Now the source. This layer rebuilds on every code change, which is cheap.
COPY src/ ./src/
COPY config/ ./config/
COPY alembic.ini ./
COPY migrations/ ./migrations/
RUN uv sync --frozen --no-dev

# The serving weights are a build artefact, not source -- git ignores models/, so a
# clean clone arrives with no brain at all. I publish them as a GitHub Release asset
# and pull them in here, at BUILD time. Render's free tier spins down when idle, so
# fetching at START time would repeat this download on every cold start and put a
# minute in front of the first request.
#
# -L is not optional: GitHub answers with a 302 to its CDN, and without it curl saves
# the redirect page as a zero-byte file and tar fails on garbage.
ARG MODEL_RELEASE=model-v1
RUN curl -fsSL -o /tmp/model.tar.gz \
    "https://github.com/Manglam11/news_vane/releases/download/${MODEL_RELEASE}/newsvane-model-${MODEL_RELEASE#model-}.tar.gz" \
    && tar -xzf /tmp/model.tar.gz -C /app \
    && rm /tmp/model.tar.gz


FROM python:3.12-slim

WORKDIR /app

# Running as root inside a container is a habit that gets punished in production.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder --chown=appuser:appuser /app /app

# The virtual environment's binaries go on PATH, so I can call uvicorn directly.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# Documentation only -- this is the port I use locally. A managed host injects its own
# PORT and routes traffic there, which is why the command below reads it at runtime.
EXPOSE 8000

# Shell form, deliberately. The JSON-array form does not run a shell, so it can neither
# chain the migration nor expand $PORT -- it would pass the literal string through.
# The image must be able to build its own schema from nothing before it answers a single
# request, and it must bind wherever its host tells it to.
CMD alembic upgrade head && uvicorn newsvane.api.main:app --host 0.0.0.0 --port ${PORT:-8000}