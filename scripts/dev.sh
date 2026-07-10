#!/usr/bin/env bash
# Dev startup: ensure Qdrant is up, then run uvicorn in the foreground.
# Qdrant is left running on exit — it's cheap and stops with `docker stop`.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

if ! curl -sf -m 2 "$QDRANT_URL/healthz" > /dev/null; then
    echo "[dev] qdrant not responding, starting container..."
    docker compose up -d qdrant
    for _ in $(seq 1 30); do
        curl -sf -m 2 "$QDRANT_URL/healthz" > /dev/null && break
        sleep 1
    done
    curl -sf -m 2 "$QDRANT_URL/healthz" > /dev/null \
        || { echo "[dev] qdrant did not come up, check: docker compose logs qdrant"; exit 1; }
fi
echo "[dev] qdrant ok at $QDRANT_URL"

exec uv run uvicorn app.api.api:app --reload
