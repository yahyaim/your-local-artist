#!/usr/bin/env python3
"""Generate a tiny synthetic test clip: a cartoon figure walking and waving.

Usage:
    backend/.venv/bin/python scripts/make_sample_clip.py [out.mp4] [--seconds 4] [--fps 24]

The figure has clear outlines, flat shaded regions and a textured background so
all three drawing styles (and background removal) have something to work on.
"""

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 360


def _background() -> np.ndarray:
    # Vertical gradient plus a few soft shapes so edge detectors see clutter.
    ys = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    top, bottom = np.array([218, 241, 241], np.float32), np.array([186, 213, 218], np.float32)
    bg = np.broadcast_to(top + (bottom - top) * ys, (HEIGHT, WIDTH, 3)).copy().astype(np.uint8)
    cv2.rectangle(bg, (40, 200), (150, 330), (150, 170, 175), -1)
    cv2.circle(bg, (560, 80), 40, (205, 225, 230), -1)
    cv2.line(bg, (0, 300), (WIDTH, 300), (130, 150, 155), 2)
    return bg


def _draw_figure(img: np.ndarray, t: float) -> None:
    ink = (40, 38, 30)
    cx = int(120 + (WIDTH - 240) * (0.5 - 0.5 * math.cos(t * math.pi / 2)))
    bob = int(4 * math.sin(t * 2 * math.pi * 2))
    hip = (cx, 230 + bob)
    neck = (cx, 150 + bob)

    # Legs swing in opposite phase.
    swing = 22 * math.sin(t * 2 * math.pi * 2)
    for sign in (1, -1):
        foot = (int(cx + sign * swing), 298)
        cv2.line(img, hip, foot, (60, 80, 140), 14, cv2.LINE_AA)
        cv2.line(img, hip, foot, ink, 2, cv2.LINE_AA)

    # Torso (filled shirt with outline).
    torso = np.array([[cx - 28, 150 + bob], [cx + 28, 150 + bob], [cx + 22, 235 + bob], [cx - 22, 235 + bob]])
    cv2.fillPoly(img, [torso], (60, 60, 230), cv2.LINE_AA)
    cv2.polylines(img, [torso], True, ink, 2, cv2.LINE_AA)

    # Left arm hangs; right arm waves.
    cv2.line(img, (cx - 26, 160 + bob), (cx - 45, 215 + bob), ink, 6, cv2.LINE_AA)
    wave = math.radians(-60 + 35 * math.sin(t * 2 * math.pi * 3))
    hand = (int(cx + 26 + 50 * math.cos(wave)), int(160 + bob + 50 * math.sin(wave)))
    cv2.line(img, (cx + 26, 160 + bob), hand, ink, 6, cv2.LINE_AA)

    # Head with simple face.
    head = (cx, neck[1] - 30)
    cv2.circle(img, head, 28, (170, 200, 240), -1, cv2.LINE_AA)
    cv2.circle(img, head, 28, ink, 2, cv2.LINE_AA)
    cv2.circle(img, (head[0] - 9, head[1] - 5), 3, ink, -1, cv2.LINE_AA)
    cv2.circle(img, (head[0] + 9, head[1] - 5), 3, ink, -1, cv2.LINE_AA)
    cv2.ellipse(img, (head[0], head[1] + 8), (10, 6), 0, 10, 170, ink, 2, cv2.LINE_AA)


def make_sample_clip(out: Path, seconds: float = 4.0, fps: int = 24) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * fps))
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    bg = _background()
    for i in range(frames):
        frame = bg.copy()
        _draw_figure(frame, i / max(frames - 1, 1) * seconds / 4.0)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the sample clip")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="storage/sample_clip.mp4", type=Path)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    print(f"Wrote {make_sample_clip(args.out, args.seconds, args.fps)}")


if __name__ == "__main__":
    main()
