FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

# the app runs unprivileged: code and venv stay root-owned and read-only to
# it. It writes only to the bind-mounted data/ and preprocessing/data/; uid
# 1000 is the usual first user on a host, who should own those folders
RUN useradd --create-home --uid 1000 app

WORKDIR /app
# bytecode up front: the app user can't write __pycache__ at runtime. copy:
# the uv cache sits in a build-cache mount, out of the image, and can't be
# hard-linked across that boundary
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies first so this layer is cached across code changes;
# --locked fails the build if uv.lock is out of date with pyproject.toml
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
USER app
EXPOSE 8000
CMD ["uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
