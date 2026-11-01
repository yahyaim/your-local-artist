"""Disk cleanup: per-frame caches go after every job, projects on "Start over" or when idle."""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

# Pencil with the background kept, so neither SAM 2 nor the line network runs.
SETTINGS = {"remove_background": False, "max_side": 320, "smoothing": 0.0}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(storage_dir=tmp_path / "storage", prefetch_models=False, project_ttl_hours=1)
    with TestClient(create_app(settings)) as c:
        yield c


def _project(client: TestClient, clip: Path) -> str:
    up = client.post("/api/uploads", params={"filename": "me.mp4"}, content=clip.read_bytes()).json()
    return client.post(f"/api/uploads/{up['upload_id']}/confirm").json()["id"]


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(600):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_job_keeps_the_render_but_not_the_frames(client: TestClient, sample_clip: Path) -> None:
    pid = _project(client, sample_clip)
    job = client.post(f"/api/projects/{pid}/jobs", json={"style": "pencil", "settings": SETTINGS}).json()
    job = _wait(client, job["id"])
    assert job["status"] == "done", job

    project_dir = client.app.state.storage.projects_dir / pid
    for name in ("frames", "masks", "lines", "flow"):
        assert not (project_dir / "cache" / name).exists()
    render = project_dir / "renders" / job["render_id"]
    assert (render / "preview.mp4").is_file() and any((render / "strokes").iterdir())


def test_start_over_deletes_the_project(client: TestClient, sample_clip: Path) -> None:
    pid = _project(client, sample_clip)
    project_dir = client.app.state.storage.projects_dir / pid
    assert project_dir.is_dir()

    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert not project_dir.exists()
    assert client.get(f"/api/projects/{pid}").status_code == 404
    assert client.delete(f"/api/projects/{pid}").status_code == 404


def test_idle_projects_are_swept_unless_busy(client: TestClient, sample_clip: Path) -> None:
    storage = client.app.state.storage
    idle, busy = _project(client, sample_clip), _project(client, sample_clip)
    later = time.time() + 3601

    assert storage.sweep_expired() == []  # just used
    assert storage.sweep_expired(now=later, busy=frozenset({busy})) == [idle]
    assert not (storage.projects_dir / idle).exists()
    assert (storage.projects_dir / busy).is_dir()


def test_using_a_project_keeps_it_alive(client: TestClient, sample_clip: Path) -> None:
    storage = client.app.state.storage
    pid = _project(client, sample_clip)
    marker = storage.projects_dir / pid / "project.json"
    old = time.time() - 7200
    os.utime(marker, (old, old))

    client.get(f"/api/projects/{pid}")  # any request on the project counts as use
    assert storage.sweep_expired(now=time.time() + 60) == []
