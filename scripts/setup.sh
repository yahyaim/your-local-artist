#!/usr/bin/env bash
# One-time (and re-runnable) setup: Python 3.11 venv + backend deps, frontend deps.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

say "Checking system tools"
command -v ffmpeg >/dev/null || die "FFmpeg not found. Install it (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg)."
command -v node >/dev/null || die "Node.js 18+ not found. Install it from https://nodejs.org or with brew install node."

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if command -v python3.11 >/dev/null; then
    PY=python3.11
  elif command -v brew >/dev/null; then
    say "Python 3.11 not found - installing with Homebrew"
    brew install python@3.11
    PY="$(brew --prefix python@3.11)/bin/python3.11"
  else
    die "Python 3.11 not found. Install it, or run with PYTHON=/path/to/python3.11 scripts/setup.sh"
  fi
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
  || die "$PY is not Python 3.11"

say "Creating backend/.venv"
[[ -d backend/.venv ]] || "$PY" -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip --quiet
backend/.venv/bin/python -m pip install -r backend/requirements.txt

say "Installing frontend dependencies"
(cd frontend && npm install --no-fund --no-audit)

mkdir -p storage/temp storage/projects
say "Done. Start the app with: make dev"
