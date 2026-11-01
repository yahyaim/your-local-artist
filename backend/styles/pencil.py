"""Sketchy pencil: XDoG edges drawn as several offset, jittered, tapered passes,
plus hatching in shadow areas.

All the "hand-made" randomness (wobble, overshoot, width, hatch ends) comes
from smooth noise fields fixed to the frame, not from a per-frame random
generator: a line in the same place gets the same wobble on every frame, and
a moving line's wobble changes gradually. Re-rolling the dice each frame would
make the drawing "boil".
"""

import math
from functools import lru_cache

import cv2
import numpy as np

from pipeline.flow import sample
from pipeline.lines import xdog
from pipeline.strokes import Material, Stroke, StrokeFrame
from pipeline.vectorize import remove_small_components, resample, sample_map

from .base import FrameInput, Maps, Style, StyleParams, centreline_paths, lerp, subject_region

GRAPHITE = "#33383b"


@lru_cache(maxsize=64)
def _field(width: int, height: int, key: int) -> np.ndarray:
    """Smooth noise in about [-1, 1] with features ~60 px (at 720 p) across."""
    cell = 60 * max(width, height) / 720
    gw, gh = max(2, int(width / cell) + 2), max(2, int(height / cell) + 2)
    coarse = np.random.default_rng(key).standard_normal((gh, gw)).astype(np.float32)
    return np.clip(cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC) / 1.5, -1, 1)


def _noise(size: tuple[int, int], key: int, points: np.ndarray, anchor: tuple[float, float]) -> np.ndarray:
    # Fields are sampled in subject-relative coordinates (see FrameInput.anchor).
    # They're padded by wrapping so a subject that moves far still has noise.
    w, h = size
    rel = points - np.asarray(anchor, np.float32)
    rel = np.column_stack([np.mod(rel[:, 0], w - 1), np.mod(rel[:, 1], h - 1)])
    return sample(_field(w, h, key), rel)


def _normals(pts: np.ndarray) -> np.ndarray:
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    return np.column_stack([-t[:, 1], t[:, 0]])


def _extend(pts: np.ndarray, start: float, end: float) -> np.ndarray:
    """Overshoot (positive) or stop short (negative) at each end, like a quick hand-drawn line."""
    if len(pts) < 3:
        return pts
    t0 = pts[0] - pts[min(2, len(pts) - 1)]
    t1 = pts[-1] - pts[max(-3, -len(pts))]
    t0 /= np.linalg.norm(t0) + 1e-6
    t1 /= np.linalg.norm(t1) + 1e-6
    out = pts
    if start > 0:
        out = np.vstack([pts[0] + t0 * start, out])
    elif start < 0:
        out = out[int(min(-start / 2.5, len(out) // 3)):]
    if end > 0:
        out = np.vstack([out, pts[-1] + t1 * end])
    elif end < 0:
        out = out[: len(out) - int(min(-end / 2.5, len(out) // 3))]
    return out.astype(np.float32)


def _sketch_passes(pts: np.ndarray, base_width: np.ndarray, closed: bool, scale: float,
                   size: tuple[int, int], anchor: tuple[float, float]) -> list[Stroke]:
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    n_passes = 2 if length > 20 * scale else 1
    ends = pts[[0, -1]]
    strokes = []
    for k in range(n_passes):
        p = pts
        if not closed:
            over = (_noise(size, 10 + k, ends, anchor) + 1) / 2  # [0, 1]
            p = _extend(pts, lerp(-3, 5, over[0]) * scale, lerp(-3, 5, over[1]) * scale)
        p = resample(p, 2.5 * scale, closed)
        if len(p) < 2:
            continue
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        u = s / max(s[-1], 1e-6)
        # Each pass drifts off the true line by a smooth, position-dependent amount.
        offset = _noise(size, 2 * k, p, anchor) * 1.1 * scale
        p = p + _normals(p) * offset[:, None]
        width_mult = 0.65 + 0.35 * (_noise(size, 30 + k, p[len(p) // 2:len(p) // 2 + 1], anchor)[0] + 1) / 2
        widths = np.interp(u, np.linspace(0, 1, len(base_width)), base_width) * width_mult
        taper = 1.0 if closed else 0.35 + 0.65 * np.sin(np.pi * u) ** 0.5
        pressure = 0.6 + 0.4 * np.sin(np.pi * u) ** 0.6
        opacity = (0.8 if k == 0 else 0.45) * pressure
        # Separate materials per pass, so temporal smoothing matches each pass
        # only with itself (and they can be styled apart in Blender).
        strokes.append(Stroke("graphite" if k == 0 else "graphite_light", p.astype(np.float32),
                              (widths * taper).astype(np.float32), opacity.astype(np.float32), closed))
    return strokes


def _hatch(mask: np.ndarray, angle_deg: float, spacing: float, width: float, opacity: float,
           scale: float, key: int, anchor: tuple[float, float]) -> list[Stroke]:
    """Parallel lines clipped to `mask`, on a grid that moves with the subject."""
    h, w = mask.shape
    if not mask.any():
        return []
    centre = (w / 2, h / 2)
    diag = int(math.ceil(math.hypot(w, h)))
    rot = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    rot[:, 2] += (diag / 2 - centre[0], diag / 2 - centre[1])
    # Shift the grid rows by the subject's motion across them (modulo the spacing).
    row_shift = float(rot[1, 0] * anchor[0] + rot[1, 1] * anchor[1]) % spacing
    rotated = cv2.warpAffine(mask.astype(np.uint8), rot, (diag, diag), flags=cv2.INTER_NEAREST)
    inv = cv2.invertAffineTransform(rot)
    min_len = 6 * scale
    strokes = []
    for y in np.arange(spacing / 2 + row_shift, diag - 1, spacing):
        row = rotated[int(y)]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row, [0]])))
        for x0, x1 in zip(edges[::2], edges[1::2]):
            if x1 - x0 < min_len:
                continue
            ends = (np.array([[x0, y, 1], [x1, y, 1]], np.float32) @ inv.T).astype(np.float32)
            # Stop a little short of the shadow edge, by an amount fixed to the location.
            short = (_noise((w, h), key, ends, anchor) + 1) / 2 * 3 * scale
            d = ends[1] - ends[0]
            d /= np.linalg.norm(d) + 1e-6
            ends = np.array([ends[0] + d * short[0], ends[1] - d * short[1]], np.float32)
            if np.dot(ends[1] - ends[0], d) < min_len:
                continue
            pts = resample(ends, 4 * scale)
            if len(pts) < 2:
                continue
            u = np.linspace(0, 1, len(pts))
            strokes.append(Stroke(
                "hatch", pts,
                np.full(len(pts), width, np.float32),
                (opacity * (0.7 + 0.3 * np.sin(np.pi * u))).astype(np.float32),
            ))
    return strokes


def rasterize(inp: FrameInput, params: StyleParams, _: object) -> Maps:
    s = inp.scale
    gray = cv2.cvtColor(inp.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    ink = xdog(gray, sigma=lerp(1.25, 0.8, params.detail) * math.sqrt(s), epsilon=lerp(0.24, 0.31, params.detail))
    region = subject_region(inp, params, 2 * s)
    if region is not None:
        ink = ink * region
    return {"ink": ink, "lum": cv2.GaussianBlur(gray, (0, 0), 3 * s)}


def vectorize(inp: FrameInput, maps: Maps, params: StyleParams, _: object) -> StrokeFrame:
    s = inp.scale
    w, h = inp.size
    binary = remove_small_components(maps["ink"] > 0.5, int(10 * s * s))
    # Half-thickness of each XDoG line, sampled along its centreline, drives stroke width.
    half_thickness = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)

    strokes: list[Stroke] = []
    for pts, closed in centreline_paths(binary, s, params.smoothing * 0.6, min_length=lerp(14, 5, params.detail) * s):
        base = np.clip(2 * sample_map(half_thickness, pts), 1.0 * s, 3.5 * s) * 0.75 * params.thickness
        strokes += _sketch_passes(pts, base.astype(np.float32), closed, s, (w, h), inp.anchor)

    # Hatching: two tone levels of the blurred luminance inside the subject.
    lum = maps["lum"]
    region = subject_region(inp, params, 2 * s)
    area = region if region is not None else np.ones_like(binary)
    vals = lum[area]
    if vals.size:
        t1 = min(float(np.percentile(vals, lerp(18, 35, params.detail))), 0.45)
        t2 = min(float(np.percentile(vals, lerp(6, 14, params.detail))), 0.28)
        erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow = cv2.erode(((lum < t1) & area).astype(np.uint8), erode)
        deep = cv2.erode(((lum < t2) & area).astype(np.uint8), erode)
        hatch_w = 1.0 * params.thickness * s
        strokes += _hatch(shadow > 0, 45, 7 * s, hatch_w, 0.4, s, 40, inp.anchor)
        strokes += _hatch(deep > 0, -45, 7 * s, hatch_w, 0.4, s, 41, inp.anchor)

    return StrokeFrame(
        width=w, height=h,
        materials=[Material("graphite", GRAPHITE), Material("graphite_light", GRAPHITE), Material("hatch", GRAPHITE)],
        strokes=strokes,
    )


STYLE = Style(
    key="pencil",
    label="Sketchy pencil",
    description="XDoG edges as several jittered, tapered passes, with hatching in the shadows.",
    needs_neural_lines=False,
    prepare=lambda frames, params: None,
    rasterize=rasterize,
    vectorize=vectorize,
    background=lambda params: "paper" if params.paper_texture else "#ffffff",
)
