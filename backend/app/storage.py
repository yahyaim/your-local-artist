"""Temp uploads and permanent project storage.

Layout:
    storage/temp/<upload_id>/source.<ext>     raw upload, not yet confirmed
    storage/temp/<upload_id>/upload.json      original name + probed metadata
    storage/projects/<project_id>/source.<ext>
    storage/projects/<project_id>/project.json

An upload only becomes a project when the user clicks "Use this video". Temp
uploads are removed on discard, or by `sweep_expired` once nothing in them has
been written for `temp_ttl_seconds` (an upload still streaming keeps bumping
its file mtime, so it is never swept mid-transfer).

Projects are deleted when the user clicks "Start over", or by `sweep_expired`
once unused for `project_ttl_seconds`. Every request that touches a project
bumps the mtime of its project.json, which is what "unused" is measured from.
"""

import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from .video_meta import VideoMetadata

ALLOWED_EXTENSIONS = (".mp4", ".mov", ".webm")
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class NotFound(Exception):
    pass


class UnsupportedFile(Exception):
    pass


class Project(BaseModel):
    id: str
    created_at: str
    original_name: str
    source_file: str
    metadata: VideoMetadata


def _new_id() -> str:
    return uuid.uuid4().hex


def _check_id(value: str) -> str:
    # IDs end up in filesystem paths, so reject anything that isn't one of ours.
    if not _ID_RE.match(value):
        raise NotFound(value)
    return value


class Storage:
    def __init__(self, temp_dir: Path, projects_dir: Path, temp_ttl_seconds: float,
                 project_ttl_seconds: float) -> None:
        self.temp_dir = temp_dir
        self.projects_dir = projects_dir
        self.temp_ttl_seconds = temp_ttl_seconds
        self.project_ttl_seconds = project_ttl_seconds
        self._lock = threading.Lock()
        temp_dir.mkdir(parents=True, exist_ok=True)
        projects_dir.mkdir(parents=True, exist_ok=True)

    # --- temp uploads -------------------------------------------------------

    def new_temp_upload(self, original_name: str) -> tuple[str, Path]:
        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFile(f"Unsupported file type '{ext or original_name}'. Use mp4, mov or webm.")
        upload_id = _new_id()
        upload_dir = self.temp_dir / upload_id
        upload_dir.mkdir(parents=True)
        (upload_dir / "upload.json").write_text(json.dumps({"original_name": Path(original_name).name}))
        return upload_id, upload_dir / f"source{ext}"

    def _upload_dir(self, upload_id: str) -> Path:
        upload_dir = self.temp_dir / _check_id(upload_id)
        if not upload_dir.is_dir():
            raise NotFound(upload_id)
        return upload_dir

    def _upload_video(self, upload_dir: Path) -> Path:
        video = next((p for p in upload_dir.glob("source.*")), None)
        if video is None:
            raise NotFound(upload_dir.name)
        return video

    def set_upload_metadata(self, upload_id: str, metadata: VideoMetadata) -> None:
        info_path = self._upload_dir(upload_id) / "upload.json"
        info = json.loads(info_path.read_text())
        info["metadata"] = metadata.model_dump()
        info_path.write_text(json.dumps(info))

    def discard(self, upload_id: str) -> None:
        with self._lock:
            shutil.rmtree(self._upload_dir(upload_id), ignore_errors=True)

    def confirm(self, upload_id: str) -> Project:
        with self._lock:
            upload_dir = self._upload_dir(upload_id)
            info = json.loads((upload_dir / "upload.json").read_text())
            if "metadata" not in info:
                raise NotFound(upload_id)  # upload never finished probing
            video = self._upload_video(upload_dir)

            project_id = _new_id()
            project_dir = self.projects_dir / project_id
            project_dir.mkdir(parents=True)
            # Same filesystem, so this is a rename rather than a copy.
            shutil.move(str(video), project_dir / video.name)
            project = Project(
                id=project_id,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                original_name=info["original_name"],
                source_file=video.name,
                metadata=VideoMetadata(**info["metadata"]),
            )
            (project_dir / "project.json").write_text(project.model_dump_json(indent=2))
            shutil.rmtree(upload_dir, ignore_errors=True)
            return project

    def sweep_expired(self, now: float | None = None, busy: frozenset[str] = frozenset()) -> list[str]:
        """Delete temp uploads with no file activity within the temp TTL, and projects
        unused for the project TTL (except `busy` ones, which have a job). Returns removed IDs."""
        now = time.time() if now is None else now
        removed: list[str] = []
        with self._lock:
            for upload_dir in self.temp_dir.iterdir():
                if not upload_dir.is_dir():
                    continue
                mtimes = [p.stat().st_mtime for p in upload_dir.iterdir()] or [upload_dir.stat().st_mtime]
                if now - max(mtimes) > self.temp_ttl_seconds:
                    shutil.rmtree(upload_dir, ignore_errors=True)
                    removed.append(upload_dir.name)
            for project_dir in self.projects_dir.iterdir():
                if not project_dir.is_dir() or project_dir.name in busy:
                    continue
                # A folder without project.json is left over from a deletion that raced a write.
                marker = project_dir / "project.json"
                last_used = (marker if marker.is_file() else project_dir).stat().st_mtime
                if now - last_used > self.project_ttl_seconds:
                    shutil.rmtree(project_dir, ignore_errors=True)
                    removed.append(project_dir.name)
        return removed

    # --- projects -----------------------------------------------------------

    def project_dir(self, project_id: str) -> Path:
        project_dir = self.projects_dir / _check_id(project_id)
        marker = project_dir / "project.json"
        try:
            os.utime(marker)  # mark as in use, for the idle sweep
        except FileNotFoundError as exc:
            raise NotFound(project_id) from exc
        return project_dir

    def delete_project(self, project_id: str) -> None:
        """Remove the project folder: source video, caches, renders and exports."""
        with self._lock:
            shutil.rmtree(self.project_dir(project_id), ignore_errors=True)

    def get_project(self, project_id: str) -> Project:
        return Project.model_validate_json((self.project_dir(project_id) / "project.json").read_text())

    def project_source(self, project_id: str) -> Path:
        project = self.get_project(project_id)
        return self.project_dir(project_id) / project.source_file
