# spatialDINO Web UI

Vite + React + TypeScript frontend.

## Dev

```bash
cd ../..
uv sync --all-packages

cd apps/web
npm install
npm run dev
```

`npm run dev` also starts the backend from the repo root workspace (via `uv run --all-packages spatialdino-server ...`) and proxies `/api/*`.
By default it prefers backend port `8000`, and if that port is busy it automatically picks the next available port.

If port `8000` is already in use, choose a different backend port:

```bash
SPATIALDINO_DEV_API_PORT=8010 npm run dev
```

The frontend proxy target can also be set directly:

```bash
SPATIALDINO_DEV_API_TARGET=http://127.0.0.1:8010 npm run dev:ui
```
