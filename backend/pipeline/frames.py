"""Frame extraction via FFmpeg."""

import subprocess
from pathlib import Path

import numpy as np


def output_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale so the long side is at most `max_side`, keeping both sides even (H.264 needs it)."""
    scale = min(1.0, max_side / max(width, height))
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


def read_frame(video: Path, time_s: float, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 frame nearest to `time_s`, scaled to `size` (w, h)."""
    w, h = size
    # -ss before -i seeks to the previous keyframe then decodes accurately to the
    # exact time. FFmpeg applies rotation metadata automatically.
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{max(time_s, 0):.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        capture_output=True, check=True,
    )
    if len(result.stdout) < w * h * 3:
        # Seeking past the last frame yields nothing: fall back to slightly earlier.
        if time_s > 0.1:
            return read_frame(video, time_s - 0.25, size)
        raise RuntimeError(f"could not decode a frame at {time_s:.2f}s")
    return np.frombuffer(result.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()
