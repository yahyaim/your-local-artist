"""SVG zip and Blender .blend exports of a finished render.

POST builds the file (once; later calls return immediately), GET downloads it.
Two steps rather than one GET so the UI can show progress and readable errors
instead of a failed browser download.
"""

import json
import re
import threading
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from export.blender import export_blend, find_blender
from export.svg import build_svg_zip

from ..deps import get_storage
from ..storage import NotFound, Storage

router = APIRouter(prefix="/api/projects/{project_id}/renders/{render_id}/exports", tags=["exports"])

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_FILES = {"svg": ("svg_frames.zip", "application/zip"), "blend": ("strokes.blend", "application/x-blender")}
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def _render_dir(storage: Storage, project_id: str, render_id: str) -> Path:
    if not _ID_RE.match(render_id):
        raise HTTPException(404, "Render not found.")
    try:
        d = storage.project_dir(project_id) / "renders" / render_id
    except NotFound as exc:
        raise HTTPException(404, "Project not found.") from exc
    if not (d / "render.json").is_file():
        raise HTTPException(404, "Render not found.")
    return d


def _kind(kind: str) -> tuple[str, str]:
    if kind not in _FILES:
        raise HTTPException(404, f"Unknown export '{kind}'.")
    return _FILES[kind]


@router.post("/{kind}")
async def build_export(project_id: str, render_id: str, kind: str, request: Request,
                       storage: Storage = Depends(get_storage)) -> dict:
    name, _ = _kind(kind)
    d = _render_dir(storage, project_id, render_id)
    out = d / name
    blender = find_blender(request.app.state.settings.blender_path)
    if kind == "blend" and not blender.available:
        raise HTTPException(503, blender.error)

    def build() -> None:
        with _locks[str(out)]:  # two clicks -> one build
            if out.is_file():
                return
            if kind == "svg":
                build_svg_zip(d / "strokes", out)
            else:
                info = json.loads((d / "render.json").read_text())
                export_blend(blender, d / "strokes", out, fps=info["fps"],
                             name=f"Strokes_{info['style']}")

    try:
        await run_in_threadpool(build)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ready": True, "size_bytes": out.stat().st_size}


@router.get("/{kind}")
def download_export(project_id: str, render_id: str, kind: str,
                    storage: Storage = Depends(get_storage)) -> FileResponse:
    name, media_type = _kind(kind)
    d = _render_dir(storage, project_id, render_id)
    if not (d / name).is_file():
        raise HTTPException(404, "Not built yet; POST to this URL first.")
    info = json.loads((d / "render.json").read_text())
    stem = Path(storage.get_project(project_id).original_name).stem
    return FileResponse(d / name, media_type=media_type, filename=f"{stem}-{info['style']}-{name}")
