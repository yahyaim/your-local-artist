"""Style-selection screen: device info, frame grabs, subject selection, style samples."""

from typing import Annotated

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from export.blender import find_blender
from pipeline import weights
from pipeline.device import device_info
from styles import STYLES, StyleParams

from ..previews import PreviewService, SubjectState
from ..storage import NotFound

router = APIRouter(prefix="/api", tags=["preview"])

DEFAULT_MAX_SIDE_GPU = 720
DEFAULT_MAX_SIDE_CPU = 480  # CPU is roughly 10x slower; start smaller

MaxSide = Annotated[int | None, Query(ge=128, le=4096)]


def get_previews(request: Request) -> PreviewService:
    return request.app.state.previews


def _max_side(value: int | None) -> int:
    if value is not None:
        return value
    return DEFAULT_MAX_SIDE_CPU if device_info().is_slow else DEFAULT_MAX_SIDE_GPU


def _png(data: bytes) -> Response:
    # Content is keyed by query params, so the browser may cache it.
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


class SubjectOut(BaseModel):
    time: float
    mode: str
    box: list[float] | None
    points: list[list[float]]
    version: str  # changes whenever the selection changes; used to bust image caches


def _subject_out(state: SubjectState) -> SubjectOut:
    return SubjectOut(**state.model_dump(), version=state.version)


class SubjectIn(BaseModel):
    time: float
    points: list[list[float]] = []  # normalised [x, y, 1=include|0=exclude]; empty = auto-detect


@router.get("/system")
def system(request: Request) -> dict:
    info = device_info()
    blender = find_blender(request.app.state.settings.blender_path)
    return {
        "blender": {"available": blender.available, "path": blender.path, "version": blender.version,
                    "api": blender.api, "error": blender.error},
        "device": {"kind": info.kind, "name": info.name, "is_slow": info.is_slow},
        # On CPU, start smaller and shorter: half frame rate and the first 5 seconds.
        "defaults": {"max_side": _max_side(None), "half_fps": info.is_slow, "max_seconds": 5 if info.is_slow else None},
        "models": weights.status(),
    }


@router.get("/styles")
def list_styles() -> list[dict[str, str]]:
    return [{"key": s.key, "label": s.label, "description": s.description} for s in STYLES.values()]


def _guard(fn):
    """Map storage and model errors to HTTP errors."""
    try:
        return fn()
    except NotFound as exc:
        raise HTTPException(404, "Project not found.") from exc
    except RuntimeError as exc:  # model download / ffmpeg failures
        raise HTTPException(503, str(exc)) from exc


@router.get("/projects/{project_id}/frame")
def frame(project_id: str, t: float = 1.0, max_side: MaxSide = None,
          previews: PreviewService = Depends(get_previews)) -> Response:
    rgb = _guard(lambda: previews.frame(project_id, t, _max_side(max_side)))
    ok, jpg = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/projects/{project_id}/subject", response_model=SubjectOut)
def get_subject(project_id: str, t: float = 1.0, max_side: MaxSide = None,
                previews: PreviewService = Depends(get_previews)) -> SubjectOut:
    return _subject_out(_guard(lambda: previews.subject(project_id, t, _max_side(max_side))))


@router.post("/projects/{project_id}/subject", response_model=SubjectOut)
def set_subject(project_id: str, body: SubjectIn, max_side: MaxSide = None,
                previews: PreviewService = Depends(get_previews)) -> SubjectOut:
    for p in body.points:
        if len(p) != 3 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) or p[2] not in (0, 1):
            raise HTTPException(422, "points must be [x, y, 1|0] with x, y in [0, 1]")
    return _subject_out(_guard(lambda: previews.set_subject(project_id, body.time, body.points, _max_side(max_side))))


@router.get("/projects/{project_id}/subject/overlay")
def subject_overlay(project_id: str, t: float = 1.0, max_side: MaxSide = None,
                    previews: PreviewService = Depends(get_previews)) -> Response:
    return _png(_guard(lambda: previews.mask_overlay_png(project_id, t, _max_side(max_side))))


@router.get("/projects/{project_id}/styles/{style}/preview")
def style_preview(project_id: str, style: str, params: Annotated[StyleParams, Query()], t: float = 1.0,
                  max_side: MaxSide = None, previews: PreviewService = Depends(get_previews)) -> Response:
    if style not in STYLES:
        raise HTTPException(404, f"Unknown style '{style}'.")
    return _png(_guard(lambda: previews.style_preview_png(project_id, style, t, _max_side(max_side), params)))
