"""Upload -> preview -> confirm/discard.

The browser sends the file as the raw request body (not multipart) so it
streams straight into the temp folder without being spooled twice, and so the
client can report upload progress with a plain XHR.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from ..deps import get_storage
from ..storage import NotFound, Project, Storage, UnsupportedFile
from ..video_meta import InvalidVideo, VideoMetadata, probe

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


class UploadOut(BaseModel):
    upload_id: str
    filename: str
    metadata: VideoMetadata


@router.post("", response_model=UploadOut, status_code=201)
async def create_upload(
    request: Request,
    filename: str = Query(..., min_length=1, max_length=255),
    storage: Storage = Depends(get_storage),
) -> UploadOut:
    try:
        upload_id, video_path = storage.new_temp_upload(filename)
    except UnsupportedFile as exc:
        raise HTTPException(415, str(exc)) from exc

    try:
        with video_path.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        metadata = await run_in_threadpool(probe, video_path)
    except ClientDisconnect:
        storage.discard(upload_id)
        raise HTTPException(400, "Upload was interrupted.") from None
    except InvalidVideo as exc:
        storage.discard(upload_id)
        raise HTTPException(422, f"Couldn't read this file as a video: {exc}") from exc
    except BaseException:
        storage.discard(upload_id)
        raise

    storage.set_upload_metadata(upload_id, metadata)
    return UploadOut(upload_id=upload_id, filename=filename, metadata=metadata)


@router.delete("/{upload_id}", status_code=204)
def discard_upload(upload_id: str, storage: Storage = Depends(get_storage)) -> Response:
    try:
        storage.discard(upload_id)
    except NotFound:
        pass  # already gone (discarded twice, or swept) - discarding is idempotent
    return Response(status_code=204)


@router.post("/{upload_id}/confirm", response_model=Project, status_code=201)
def confirm_upload(upload_id: str, storage: Storage = Depends(get_storage)) -> Project:
    try:
        return storage.confirm(upload_id)
    except NotFound as exc:
        raise HTTPException(404, "Upload not found or expired. Please upload the video again.") from exc
