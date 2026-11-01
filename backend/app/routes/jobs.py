"""Generation jobs (start, progress via Server-Sent Events, cancel) and render outputs."""

import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from pipeline.generate import GenerateSettings, SubjectSpec, time_range
from styles import STYLES

from ..deps import get_storage
from ..jobs import JobManager
from ..previews import PreviewService
from ..storage import NotFound, Storage

router = APIRouter(prefix="/api", tags=["jobs"])

_RENDER_FILES = {"preview.mp4", "original.mp4"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_previews(request: Request) -> PreviewService:
    return request.app.state.previews


class JobIn(BaseModel):
    style: str
    settings: GenerateSettings = GenerateSettings()


@router.post("/projects/{project_id}/jobs", status_code=201)
def start_job(project_id: str, body: JobIn, jobs: JobManager = Depends(get_jobs),
              storage: Storage = Depends(get_storage), previews: PreviewService = Depends(get_previews)) -> dict:
    if body.style not in STYLES:
        raise HTTPException(422, f"Unknown style '{body.style}'.")
    try:
        project = storage.get_project(project_id)
        meta = project.metadata
        time_range(meta.duration_s, meta.fps / (2 if body.settings.half_fps else 1), body.settings)
        # The subject picked on the style screen; auto-detect at the range start if none yet.
        state = previews.saved_subject(project_id)
    except NotFound as exc:
        raise HTTPException(404, "Project not found.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    subject = (
        SubjectSpec(time=state.time, box=state.box, points=state.points, version=state.version)
        if state else SubjectSpec(time=-1.0, box=None, version="auto")
    )
    return jobs.submit(project_id, body.style, body.settings, subject).snapshot()


def _job(job_id: str, jobs: JobManager):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (the backend may have restarted).")
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str, jobs: JobManager = Depends(get_jobs)) -> dict:
    return _job(job_id, jobs).snapshot()


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str, jobs: JobManager = Depends(get_jobs)) -> dict:
    _job(job_id, jobs)
    jobs.cancel(job_id)
    return {"ok": True}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, jobs: JobManager = Depends(get_jobs)) -> StreamingResponse:
    job = _job(job_id, jobs)

    async def stream():
        last = None
        while not await request.is_disconnected():
            snap = job.snapshot()
            # ETA and elapsed tick every second; only send when something else changed
            # or at most once a second, to keep the stream light.
            key = json.dumps({**snap, "eta_s": None, "elapsed_s": int(snap["elapsed_s"])})
            if key != last:
                last = key
                yield f"data: {json.dumps(snap)}\n\n"
            if snap["status"] in ("done", "failed", "cancelled"):
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


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


@router.get("/projects/{project_id}/renders/{render_id}")
def get_render(project_id: str, render_id: str, storage: Storage = Depends(get_storage)) -> dict:
    return json.loads((_render_dir(storage, project_id, render_id) / "render.json").read_text())


@router.get("/projects/{project_id}/renders/{render_id}/{name}")
def get_render_file(project_id: str, render_id: str, name: str, download: bool = False,
                    storage: Storage = Depends(get_storage)) -> FileResponse:
    if name not in _RENDER_FILES:
        raise HTTPException(404, "File not found.")
    d = _render_dir(storage, project_id, render_id)
    filename = None
    if download:
        info = json.loads((d / "render.json").read_text())
        stem = Path(storage.get_project(project_id).original_name).stem
        filename = f"{stem}-{info['style']}-{name}"
    return FileResponse(d / name, media_type="video/mp4", filename=filename)
