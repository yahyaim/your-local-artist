"""Find Blender and run the headless Grease Pencil exporter (blender_export/build_grease_pencil.py)."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "blender_export" / "build_grease_pencil.py"

# Checked after BLENDER_PATH / the config setting and PATH: the default install locations.
_STANDARD_LOCATIONS = {
    "darwin": ["/Applications/Blender.app/Contents/MacOS/Blender"],
    "linux": ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"],
    "win32": [str(p) for p in sorted(Path("C:/Program Files/Blender Foundation").glob("Blender*/blender.exe"), reverse=True)],
}

HOW_TO_FIX = (
    "Install Blender 4.x from blender.org, or point the app at it: add a line like "
    "BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender to the .env file in the "
    "project folder (or export BLENDER_PATH before `make dev`), then restart the app."
)


@dataclass(frozen=True)
class BlenderInfo:
    available: bool
    path: str | None = None
    version: str | None = None
    api: str | None = None  # "GPv3" (Blender 4.3+) or "GPv2" (4.0-4.2)
    error: str | None = None


def _version(path: str) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"Blender (\d+)\.(\d+)(?:\.(\d+))?", out)
    return (int(m[1]), int(m[2]), int(m[3] or 0)) if m else None


@cache
def find_blender(configured: str | None) -> BlenderInfo:
    """Locate Blender: config/BLENDER_PATH first, then PATH, then standard install folders.

    Cached per configured value; restart the app after installing Blender.
    """
    if configured:
        candidates = [configured]
    else:
        candidates = [p for p in [shutil.which("blender")] if p] + _STANDARD_LOCATIONS.get(sys.platform, [])
    for path in candidates:
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            continue
        version = _version(path)
        if version is None:
            return BlenderInfo(False, path, error=f"{path} didn't respond to --version; is it Blender? {HOW_TO_FIX}")
        text = ".".join(map(str, version))
        if version < (4, 0, 0):
            return BlenderInfo(False, path, text, error=f"Blender {text} is too old; 4.0 or newer is needed. {HOW_TO_FIX}")
        return BlenderInfo(True, path, text, "GPv3" if version >= (4, 3, 0) else "GPv2")
    if configured:
        return BlenderInfo(False, configured, error=f"BLENDER_PATH is set to {configured}, but no executable is there. {HOW_TO_FIX}")
    return BlenderInfo(False, error=f"Blender wasn't found. {HOW_TO_FIX}")


def export_blend(blender: BlenderInfo, strokes_dir: Path, out: Path, fps: float, name: str = "Strokes") -> Path:
    """Run Blender headless to build `out`. Raises RuntimeError with Blender's output on failure."""
    if not blender.available or blender.path is None:
        raise RuntimeError(blender.error or "Blender is not available.")
    tmp = out.with_name(out.stem + ".part.blend")
    cmd = [
        blender.path, "-b", "--factory-startup", "-P", str(SCRIPT), "--",
        "--strokes", str(strokes_dir), "--output", str(tmp), "--fps", f"{fps:.4f}", "--name", name,
    ]
    # Blender stores the home folder (the file browser's default directory) inside
    # every .blend it saves, which would leak the username into shared files. An
    # empty throwaway home keeps it out; --factory-startup means no user config is needed.
    with tempfile.TemporaryDirectory(prefix="blender-home-", dir="/tmp" if os.path.isdir("/tmp") else None) as home:
        env = {**os.environ, "HOME": home, "USERPROFILE": home}
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    if result.returncode != 0 or not tmp.is_file():
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"Blender export failed (exit {result.returncode}):\n{tail}")
    tmp.rename(out)
    tmp.with_suffix(".blend1").unlink(missing_ok=True)  # Blender's backup copy, if any
    return out
