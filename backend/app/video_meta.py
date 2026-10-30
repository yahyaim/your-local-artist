"""Video metadata via ffprobe."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel


class InvalidVideo(Exception):
    """The file could not be read as a video with at least one video stream."""


class VideoMetadata(BaseModel):
    # Display dimensions, i.e. after applying any rotation tag (phone videos).
    width: int
    height: int
    fps: float
    duration_s: float
    size_bytes: int
    codec: str
    has_audio: bool


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    # Newer ffmpeg reports rotation in side data; older files use a `rotate` tag.
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(side["rotation"])
    return int(stream.get("tags", {}).get("rotate", 0))


def probe(path: Path) -> VideoMetadata:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True, text=True, check=True, timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found; install FFmpeg (brew install ffmpeg)") from exc
    except subprocess.CalledProcessError as exc:
        raise InvalidVideo(exc.stderr.strip() or "ffprobe failed") from exc

    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise InvalidVideo("no video stream found")

    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if abs(_rotation(video)) % 180 == 90:
        width, height = height, width

    # avg_frame_rate is the true average for variable-frame-rate phone footage;
    # r_frame_rate is the container's nominal rate and the better fallback.
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    if width <= 0 or height <= 0 or fps <= 0:
        raise InvalidVideo("could not determine resolution or frame rate")

    return VideoMetadata(
        width=width,
        height=height,
        fps=round(fps, 3),
        duration_s=round(duration, 3),
        size_bytes=path.stat().st_size,
        codec=str(video.get("codec_name", "unknown")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )
