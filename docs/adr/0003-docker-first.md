# ADR-0003: Docker first, Kubernetes later

**Decision.** Everything runs via `docker compose` (api + qdrant + ollama).
Dependencies are managed with uv (`pyproject.toml` + `uv.lock`); the image
installs with `uv sync --frozen`.

**Why.** One command brings the whole stack up reproducibly. Kubernetes is
planned for scaling; compose services map 1:1 onto future k8s Deployments,
and all wiring is already env-var based (`config/settings.py`), which is
exactly what k8s ConfigMaps/Secrets expect.

**Consequence.** No k8s manifests yet — write them when scaling is actually
needed.
