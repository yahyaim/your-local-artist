"""FFmpeg helpers for whole-clip work: frame extraction and encoding."""

import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def extract_frames(
    video: Path, out_dir: Path, start_s: float, end_s: float, fps: float, size: tuple[int, int],
    on_progress: Callable[[int], None], check_cancel: Callable[[], None],
) -> list[Path]:
    """Decode [start_s, end_s) at `fps`, scaled to `size`, into out_dir/00000.jpg ...

    JPEG because SAM 2's video predictor reads a folder of JPEG frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostats", "-progress", "pipe:1", "-y",
            "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(video),
            "-vf", f"fps={fps:.6f},scale={w}:{h}:flags=area",
            "-q:v", "2", "-start_number", "0", str(out_dir / "%05d.jpg"),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if line.startswith("frame="):
                on_progress(int(line.split("=", 1)[1]))
                check_cancel()
    except BaseException:
        proc.kill()
        raise
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr.read() if proc.stderr else ''}")
    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError("No frames in the selected time range.")
    return frames


class Encoder:
    """Pipe RGB frames into an H.264 MP4, optionally muxing audio from a source clip."""

    def __init__(self, out: Path, size: tuple[int, int], fps: float,
                 audio_from: Path | None = None, audio_range: tuple[float, float] | None = None) -> None:
        w, h = size
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
        if audio_from is not None and audio_range is not None:
            cmd += ["-ss", f"{audio_range[0]:.3f}", "-to", f"{audio_range[1]:.3f}", "-i", str(audio_from),
                    "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        # No metadata from the source (phone GPS location, device, creation time, title).
        cmd += ["-map_metadata", "-1", "-map_chapters", "-1",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            err = self.proc.stderr.read().decode() if self.proc.stderr else ""
            raise RuntimeError(f"ffmpeg encoding failed: {err}")

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()


def encode_images(images: Iterable[np.ndarray], out: Path, size: tuple[int, int], fps: float, **audio) -> None:
    enc = Encoder(out, size, fps, **audio)
    try:
        for img in images:
            enc.write(img)
    except BaseException:
        enc.abort()
        raise
    enc.close()
