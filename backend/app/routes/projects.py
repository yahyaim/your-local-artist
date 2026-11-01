from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

from ..deps import get_storage
from ..jobs import JobManager
from ..storage import NotFound, Project, Storage

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: str, storage: Storage = Depends(get_storage)) -> Project:
    try:
        return storage.get_project(project_id)
    except NotFound as exc:
        raise HTTPException(404, "Project not found.") from exc


@router.get("/{project_id}/source")
def get_project_source(project_id: str, storage: Storage = Depends(get_storage)) -> FileResponse:
    try:
        path = storage.project_source(project_id)
    except NotFound as exc:
        raise HTTPException(404, "Project not found.") from exc
    # FileResponse honours Range requests, which <video> needs for seeking.
    return FileResponse(path)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request, storage: Storage = Depends(get_storage)) -> Response:
    """"Start over": delete the project and everything generated from it."""
    jobs: JobManager = request.app.state.jobs
    if not jobs.cancel_project(project_id):
        raise HTTPException(409, "A job for this project is still stopping; try again in a moment.")
    try:
        storage.delete_project(project_id)
    except NotFound as exc:
        raise HTTPException(404, "Project not found.") from exc
    return Response(status_code=204)
