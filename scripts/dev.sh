#!/usr/bin/env bash
# Start backend (http://127.0.0.1:8000) and frontend (http://localhost:5173) together.
# Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -x backend/.venv/bin/python ]] || { echo "Run 'make setup' first." >&2; exit 1; }

trap 'kill 0' EXIT INT TERM

(cd backend && .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app --reload-dir pipeline --reload-dir styles) &
(cd frontend && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort) &

wait
