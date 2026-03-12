# spatialDINO Server

FastAPI server that exposes the `/api/*` endpoints and serves the built web UI.

The server is part of the repository's root `uv` workspace. It is meant to share the root `.venv`
with the core `spatialdino` package rather than maintaining a second environment under `apps/server/`.

## Install / run

```bash
cd /path/to/spatialdino
uv sync --all-packages
uv run --all-packages spatialdino-server --host 0.0.0.0 --port 8000
```

## Tests

```bash
cd /path/to/spatialdino
uv run --all-packages python -m unittest discover -s apps/server/tests -p "test_*.py" -v
```

## Web assets

- Default: serves `apps/web/dist/`
- Override: set `SPATIALDINO_DIST_DIR=/absolute/path/to/dist`
