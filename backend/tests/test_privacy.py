"""Shared outputs must not carry personal data from the source file or this machine."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from export.blender import export_blend, find_blender
from pipeline.strokes import Material, Stroke, StrokeFrame
from pipeline.video import encode_images


def _tags(path: Path) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", str(path)],
                          capture_output=True, text=True, check=True).stdout


def test_videos_drop_source_metadata(tmp_path: Path) -> None:
    src = tmp_path / "phone.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=12:duration=1",
        "-f", "lavfi", "-i", "sine=duration=1", "-shortest", "-c:v", "libx264", "-c:a", "aac",
        "-metadata", "location=+48.8584+002.2945/", "-metadata", "title=private",
        "-metadata", "creation_time=2026-01-01T00:00:00Z", str(src),
    ], check=True)
    assert "location" in _tags(src)

    out = tmp_path / "out.mp4"
    frames = (np.zeros((90, 160, 3), np.uint8) for _ in range(12))
    encode_images(frames, out, (160, 90), 12.0, audio_from=src, audio_range=(0.0, 1.0))
    tags = _tags(out)
    assert "location" not in tags and "private" not in tags and "creation_time" not in tags


BLENDER = find_blender(None)


@pytest.mark.skipif(not BLENDER.available or shutil.which("zstd") is None, reason="needs Blender and zstd")
def test_blend_does_not_contain_home_folder(tmp_path: Path) -> None:
    strokes = tmp_path / "strokes"
    strokes.mkdir()
    pts = np.array([[10, 10], [100, 60]], np.float32)
    frame = StrokeFrame(160, 90, [Material("ink", "#000000")],
                        strokes=[Stroke("ink", pts, np.full(2, 2, np.float32), np.ones(2, np.float32))])
    (strokes / "00000.json").write_text(json.dumps(frame.to_json()))
    out = export_blend(BLENDER, strokes, tmp_path / "out.blend", fps=12)

    raw = subprocess.run(["zstd", "-dc", str(out)], capture_output=True, check=True).stdout
    assert raw.startswith(b"BLENDER")
    assert str(Path.home()).encode() not in raw
