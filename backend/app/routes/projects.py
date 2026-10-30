from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..deps import get_storage
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
