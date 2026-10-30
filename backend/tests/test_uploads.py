import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(storage_dir=tmp_path / "storage", temp_ttl_minutes=1)
    with TestClient(create_app(settings)) as c:
        yield c


def _upload(client: TestClient, clip: Path, name: str = "me.mp4"):
    return client.post("/api/uploads", params={"filename": name}, content=clip.read_bytes())


def _storage(client: TestClient):
    return client.app.state.storage


def test_upload_probes_metadata_and_stays_in_temp(client: TestClient, sample_clip: Path) -> None:
    res = _upload(client, sample_clip)
    assert res.status_code == 201, res.text
    body = res.json()
    meta = body["metadata"]
    assert (meta["width"], meta["height"]) == (640, 360)
    assert meta["fps"] == pytest.approx(12, abs=0.01)
    assert meta["duration_s"] == pytest.approx(1.0, abs=0.1)
    assert meta["size_bytes"] == sample_clip.stat().st_size

    storage = _storage(client)
    assert (storage.temp_dir / body["upload_id"]).is_dir()
    assert not any(storage.projects_dir.iterdir())


def test_confirm_moves_upload_into_project(client: TestClient, sample_clip: Path) -> None:
    upload_id = _upload(client, sample_clip).json()["upload_id"]
    res = client.post(f"/api/uploads/{upload_id}/confirm")
    assert res.status_code == 201, res.text
    project = res.json()

    storage = _storage(client)
    assert not (storage.temp_dir / upload_id).exists()
    assert (storage.projects_dir / project["id"] / "source.mp4").is_file()
    assert client.get(f"/api/projects/{project['id']}").json()["original_name"] == "me.mp4"

    src = client.get(f"/api/projects/{project['id']}/source", headers={"Range": "bytes=0-99"})
    assert src.status_code == 206
    assert len(src.content) == 100


def test_discard_removes_temp_upload(client: TestClient, sample_clip: Path) -> None:
    upload_id = _upload(client, sample_clip).json()["upload_id"]
    assert client.delete(f"/api/uploads/{upload_id}").status_code == 204
    assert not (_storage(client).temp_dir / upload_id).exists()
    # Confirming a discarded upload fails cleanly; discarding again is a no-op.
    assert client.post(f"/api/uploads/{upload_id}/confirm").status_code == 404
    assert client.delete(f"/api/uploads/{upload_id}").status_code == 204


def test_rejects_wrong_extension_and_non_video(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/api/uploads", params={"filename": "x.gif"}, content=b"GIF89a").status_code == 415

    res = client.post("/api/uploads", params={"filename": "fake.mp4"}, content=b"not a video")
    assert res.status_code == 422
    assert not any(_storage(client).temp_dir.iterdir())


def test_sweep_removes_only_stale_uploads(client: TestClient, sample_clip: Path) -> None:
    upload_id = _upload(client, sample_clip).json()["upload_id"]
    storage = _storage(client)
    assert storage.sweep_expired() == []
    assert storage.sweep_expired(now=time.time() + 61) == [upload_id]
    assert not (storage.temp_dir / upload_id).exists()


def test_bad_ids_are_not_found(client: TestClient) -> None:
    assert client.get("/api/projects/..%2F..%2Fetc").status_code == 404
    assert client.post("/api/uploads/not-an-id/confirm").status_code == 404
