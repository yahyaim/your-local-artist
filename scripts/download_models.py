#!/usr/bin/env python3
"""Download all model weights into backend/models/ (the app also does this on first use).

Usage: backend/.venv/bin/python scripts/download_models.py [key ...]
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pipeline import weights  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = sys.argv[1:] or list(weights.WEIGHTS)
    failed = []
    for key in keys:
        try:
            print(f"{weights.WEIGHTS[key].label}: {weights.ensure(key)}")
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            failed.append(key)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}. Re-run this script to retry.")


if __name__ == "__main__":
    main()
