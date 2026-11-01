"""Full-video generation: frames -> masks -> lines -> vector strokes -> (smoothing) -> encode.

Expensive intermediates are written per project under content-derived keys:

    cache/frames/<key>/00000.jpg   key: time range, fps, size
    cache/masks/<key>/00000.png    key: frames + subject selection + SAM variant
    cache/lines/<key>/00000.png    key: masks + background removal
    cache/flow/<key>/00001.npy     key: frames + RAFT variant (backward optical flow)
Each cache folder gets a `.complete` marker only once fully written, so an
interrupted or cancelled run never leaves a half-filled cache that looks valid.
The job manager calls `clear_intermediates` after every run to free the disk
space, so a later run with another style recomputes them; `generate` itself
reuses any complete folder it finds.

Temporal stabilisation (see temporal.py) happens in two places: raster maps
and masks are smoothed along the flow while vectorising, then stroke points are
smoothed against the previous frame in the "smoothing" stage.

Outputs go to renders/<render_id>/: strokes/00000.json (the vector source of
truth), preview.mp4 (rendered from that JSON), original.mp4 (the source at the
same range/fps/size, for side-by-side playback) and render.json.
"""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from pydantic import Field

from styles import STYLES, FrameInput, StyleParams
from styles.base import neural_input

from . import lines
from .device import free_memory
from .flow import compute_backward_flows, load_flow, raft_variant, release_models, validity
from .frames import output_size
from .masks import propagate_masks
from .render import composite, render
from .strokes import StrokeFrame
from .stylize import background_image
from .subject import Prompt, auto_prompt, gpu_lock, sam2_variant
from .temporal import RasterStabilizer, smooth_strokes
from .video import encode_images, extract_frames

STAGES: list[tuple[str, str]] = [
    ("frames", "Frame extraction"),
    ("masking", "Subject masking"),
    ("lines", "Line extraction"),
    ("flow", "Optical flow"),
    ("vectorize", "Vectorization"),
    ("smoothing", "Temporal smoothing"),
    ("encoding", "Encoding"),
]


class GenerateSettings(StyleParams):
    max_side: int = Field(720, ge=128, le=4096, description="Longest output side in pixels")
    half_fps: bool = False
    start_s: float = Field(0.0, ge=0)
    end_s: float | None = Field(None, ge=0, description="None = until the end")
    low_memory: bool = False


@dataclass
class SubjectSpec:
    """The user's subject selection, in normalised [0, 1] frame coordinates."""

    time: float
    box: list[float] | None
    points: list[list[float]] = field(default_factory=list)
    version: str = ""

    def to_prompt(self, w: int, h: int) -> Prompt:
        box = [self.box[0] * w, self.box[1] * h, self.box[2] * w, self.box[3] * h] if self.box else None
        return Prompt(box=box, points=[[x * w, y * h, lab] for x, y, lab in self.points])


class Cancelled(Exception):
    pass


class Reporter(Protocol):
    def begin(self, stage: str, total: int) -> None: ...
    def advance(self, done: int) -> None: ...
    def finish(self, stage: str, status: str = "done") -> None: ...  # done | cached | skipped
    def check_cancel(self) -> None: ...


def _key(*parts: object) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _complete(d: Path) -> bool:
    return (d / ".complete").is_file()


INTERMEDIATE_CACHES = ("frames", "masks", "lines", "flow")


def clear_intermediates(project_dir: Path) -> None:
    """Delete the per-frame caches (frames, masks, line maps, flow); renders are kept."""
    for name in INTERMEDIATE_CACHES:
        shutil.rmtree(project_dir / "cache" / name, ignore_errors=True)


def _start_cache(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)


def _read_rgb(path: Path) -> np.ndarray:
    return cv2.imread(str(path))[..., ::-1].copy()


def _read_gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def time_range(duration: float, fps: float, settings: GenerateSettings) -> tuple[float, float]:
    start = min(max(settings.start_s, 0.0), duration)
    end = min(settings.end_s if settings.end_s is not None else duration, duration)
    if end - start < 1.0 / fps:
        raise ValueError("The time range is shorter than one frame.")
    return start, end


def generate(
    *, project_dir: Path, source: Path, metadata: dict, subject: SubjectSpec, style_key: str,
    settings: GenerateSettings, render_id: str, rep: Reporter,
) -> dict:
    style = STYLES[style_key]
    fps = metadata["fps"] / (2 if settings.half_fps else 1)
    start, end = time_range(metadata["duration_s"], fps, settings)
    size = output_size(metadata["width"], metadata["height"], settings.max_side)
    w, h = size
    cache = project_dir / "cache"
    started = time.time()

    # 1. Frames ------------------------------------------------------------------
    frames_key = _key(metadata["size_bytes"], start, end, round(fps, 4), size)
    frames_dir = cache / "frames" / frames_key
    if _complete(frames_dir):
        rep.finish("frames", "cached")
    else:
        _start_cache(frames_dir)
        rep.begin("frames", max(1, int((end - start) * fps)))
        extract_frames(source, frames_dir, start, end, fps, size, rep.advance, rep.check_cancel)
        (frames_dir / ".complete").touch()
        rep.finish("frames")
    frames = sorted(frames_dir.glob("*.jpg"))
    n = len(frames)

    # 2. Subject masks -------------------------------------------------------------
    masks_dir: Path | None = None
    if settings.remove_background:
        variant = sam2_variant(settings.low_memory)
        masks_dir = cache / "masks" / _key(frames_key, subject.version, variant)
        if _complete(masks_dir):
            rep.finish("masking", "cached")
        else:
            if start <= subject.time <= end:
                prompt_index = min(max(int(round((subject.time - start) * fps)), 0), n - 1)
                prompt = subject.to_prompt(w, h)
            else:
                # The subject was picked on a frame outside the range: re-detect on the first frame.
                prompt_index, prompt = 0, auto_prompt(_read_rgb(frames[0]))
            _start_cache(masks_dir)
            rep.begin("masking", n)
            propagate_masks(frames, prompt_index, prompt, masks_dir, variant, settings.low_memory,
                            rep.advance, rep.check_cancel)
            (masks_dir / ".complete").touch()
            rep.finish("masking")
    else:
        rep.finish("masking", "skipped")

    def mask_at(i: int) -> np.ndarray | None:
        return _read_gray(masks_dir / f"{i:05d}.png") > 127 if masks_dir else None

    # 3. Neural line maps ------------------------------------------------------------
    lines_dir: Path | None = None
    if style.needs_neural_lines:
        lines_dir = cache / "lines" / _key(masks_dir.name if masks_dir else frames_key, settings.remove_background)
        if _complete(lines_dir):
            rep.finish("lines", "cached")
        else:
            _start_cache(lines_dir)
            rep.begin("lines", n)
            batch = 1 if settings.low_memory else 4
            for i0 in range(0, n, batch):
                rep.check_cancel()
                idx = range(i0, min(i0 + batch, n))
                inputs = np.stack([
                    neural_input(_read_rgb(frames[i]), mask_at(i), settings.remove_background) for i in idx
                ])
                with gpu_lock:
                    inks = lines.neural_lines(inputs)
                for i, ink in zip(idx, inks):
                    # 8-bit PNG: ~100x smaller than float maps, and plenty of precision for thresholding.
                    cv2.imwrite(str(lines_dir / f"{i:05d}.png"), np.rint(ink * 255).astype(np.uint8))
                rep.advance(idx.stop)
            (lines_dir / ".complete").touch()
            if settings.low_memory:
                lines.release_line_model()
            free_memory()
            rep.finish("lines")
    else:
        rep.finish("lines", "skipped")

    def frame_input(i: int) -> FrameInput:
        inp = FrameInput(rgb=_read_rgb(frames[i]), mask=mask_at(i))
        if lines_dir is not None:
            inp.neural_ink = _read_gray(lines_dir / f"{i:05d}.png").astype(np.float32) / 255
        return inp

    # 4. Optical flow (only needed for temporal smoothing) ------------------------------
    strength = settings.smoothing
    flow_dir: Path | None = None
    if strength > 0 and n > 1:
        variant = raft_variant(settings.low_memory)
        flow_dir = cache / "flow" / _key(frames_key, variant)
        if _complete(flow_dir):
            rep.finish("flow", "cached")
        else:
            _start_cache(flow_dir)
            rep.begin("flow", n)
            compute_backward_flows(frames, flow_dir, variant, 1 if settings.low_memory else 4,
                                   rep.advance, rep.check_cancel)
            (flow_dir / ".complete").touch()
            if settings.low_memory:
                release_models()
            free_memory()
            rep.finish("flow")
    else:
        rep.finish("flow", "skipped")

    def motion(i: int, rgb: np.ndarray, prev_rgb: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if flow_dir is None or i == 0 or prev_rgb is None:
            return None, None
        flow = load_flow(flow_dir, i, size)
        return flow, validity(rgb, prev_rgb, flow)

    # 5. Vectorize, with masks and raster maps stabilised along the flow -------------
    render_dir = project_dir / "renders" / render_id
    strokes_dir = render_dir / "strokes"
    strokes_dir.mkdir(parents=True)
    rep.begin("vectorize", n)
    # Per-video style state (e.g. the rotoscope palette) from a spread of frames.
    samples = [frame_input(int(i)) for i in np.linspace(0, n - 1, min(8, n)).round()]
    state = style.prepare(samples, settings)
    # Masks only get a light touch: SAM 2 is already temporally consistent, and a
    # heavily smoothed mask lags behind fast limbs and clips their lines.
    mask_filter = RasterStabilizer(0.4 * strength)
    map_filter = RasterStabilizer(0.85 * strength)
    prev_rgb = None
    anchor = np.zeros(2, np.float32)
    for i in range(n):
        rep.check_cancel()
        inp = frame_input(i)
        flow, valid = motion(i, inp.rgb, prev_rgb)
        if inp.mask is not None:
            inp.mask = mask_filter({"mask": inp.mask.astype(np.float32)}, flow, valid)["mask"] > 0.5
        if flow is not None:
            # Backward flow points to where content was, so the subject moved by -flow.
            region = inp.mask if inp.mask is not None and inp.mask.any() else slice(None)
            anchor -= np.median(flow[region].reshape(-1, 2), axis=0)
            inp.anchor = (float(anchor[0]), float(anchor[1]))
        maps = map_filter(style.rasterize(inp, settings, state), flow, valid)
        frame = style.vectorize(inp, maps, settings, state)
        (strokes_dir / f"{i:05d}.json").write_text(json.dumps(frame.to_json(), separators=(",", ":")))
        prev_rgb = inp.rgb
        rep.advance(i + 1)
    rep.finish("vectorize")

    # 6. Temporal smoothing of the stroke points themselves ---------------------------
    def load_frame(i: int) -> StrokeFrame:
        return StrokeFrame.from_json(json.loads((strokes_dir / f"{i:05d}.json").read_text()))

    if flow_dir is not None:
        rep.begin("smoothing", n)
        prev, prev_rgb = load_frame(0), _read_rgb(frames[0])
        radius = 3.0 * max(w, h) / 720  # how far a point may be matched to last frame's line
        for i in range(1, n):
            rep.check_cancel()
            rgb = _read_rgb(frames[i])
            flow, valid = motion(i, rgb, prev_rgb)
            prev = smooth_strokes(prev, load_frame(i), flow, valid, 0.9 * strength, radius)
            (strokes_dir / f"{i:05d}.json").write_text(json.dumps(prev.to_json(), separators=(",", ":")))
            prev_rgb = rgb
            rep.advance(i + 1)
        rep.finish("smoothing")
    else:
        rep.finish("smoothing", "skipped")

    # 7. Encode ----------------------------------------------------------------------
    rep.begin("encoding", 2 * n)
    background = background_image(style_key, settings, w, h)
    audio = {"audio_from": source, "audio_range": (start, end)} if metadata.get("has_audio") else {}

    def stroke_images():
        # Rendered from the saved JSON, so the preview shows exactly what the exporters will read.
        for i in range(n):
            rep.check_cancel()
            yield composite(render(load_frame(i)), background)
            rep.advance(i + 1)

    def original_images():
        for i in range(n):
            rep.check_cancel()
            yield _read_rgb(frames[i])
            rep.advance(n + i + 1)

    encode_images(stroke_images(), render_dir / "preview.mp4", size, fps, **audio)
    encode_images(original_images(), render_dir / "original.mp4", size, fps, **audio)
    rep.finish("encoding")

    info = {
        "id": render_id,
        "style": style_key,
        "settings": settings.model_dump(),
        "subject_version": subject.version,
        "width": w, "height": h, "fps": round(fps, 4), "frames": n,
        "start_s": start, "end_s": end,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds_taken": round(time.time() - started, 1),
    }
    (render_dir / "render.json").write_text(json.dumps(info, indent=2))
    return info
