"""End-to-end generation without model weights: pencil style, background kept
(so neither SAM 2 nor the line network runs)."""

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.generate import Cancelled, GenerateSettings, SubjectSpec, generate
from pipeline.strokes import StrokeFrame


class Recorder:
    def __init__(self, cancel_at_stage: str | None = None) -> None:
        self.events: list[tuple[str, str]] = []
        self.cancel_at_stage = cancel_at_stage
        self.current = ""

    def begin(self, stage: str, total: int) -> None:
        self.current = stage
        self.events.append((stage, "begin"))

    def advance(self, done: int) -> None:
        pass

    def finish(self, stage: str, status: str = "done") -> None:
        self.events.append((stage, status))

    def check_cancel(self) -> None:
        if self.current == self.cancel_at_stage:
            raise Cancelled()


METADATA = {"width": 640, "height": 360, "fps": 12.0, "duration_s": 1.0, "size_bytes": 1, "has_audio": False}
SETTINGS = GenerateSettings(remove_background=False, max_side=320)


def _run(tmp_path: Path, clip: Path, rep: Recorder, render_id: str = "r1") -> dict:
    return generate(
        project_dir=tmp_path, source=clip, metadata=METADATA,
        subject=SubjectSpec(time=0.5, box=None, version="none"), style_key="pencil",
        settings=SETTINGS, render_id=render_id, rep=rep,
    )


def test_generate_writes_strokes_and_videos(tmp_path: Path, sample_clip: Path) -> None:
    rep = Recorder()
    info = _run(tmp_path, sample_clip, rep)

    assert info["frames"] == 12 and (info["width"], info["height"]) == (320, 180)
    render = tmp_path / "renders" / "r1"
    jsons = sorted((render / "strokes").glob("*.json"))
    assert len(jsons) == 12
    frame = StrokeFrame.from_json(json.loads(jsons[5].read_text()))
    assert frame.width == 320 and len(frame.strokes) > 0

    for name in ("preview.mp4", "original.mp4"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames,width", "-of", "csv=p=0", str(render / name)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert out == "320,12"
    assert ("masking", "skipped") in rep.events and ("lines", "skipped") in rep.events


def test_second_run_reuses_cached_frames(tmp_path: Path, sample_clip: Path) -> None:
    _run(tmp_path, sample_clip, Recorder(), "r1")
    rep = Recorder()
    _run(tmp_path, sample_clip, rep, "r2")
    assert ("frames", "cached") in rep.events


def test_cancel_leaves_no_complete_marker(tmp_path: Path, sample_clip: Path) -> None:
    with pytest.raises(Cancelled):
        _run(tmp_path, sample_clip, Recorder(cancel_at_stage="frames"))
    frame_caches = list((tmp_path / "cache" / "frames").iterdir())
    assert frame_caches and not any((d / ".complete").exists() for d in frame_caches)
