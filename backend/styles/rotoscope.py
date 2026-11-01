"""Rotoscope: flat colour regions (k-means inside the subject) with clean outlines on top.

The palette is computed once per video in `prepare` and reused for every
frame, so a region can't jump between colours from one frame to the next.
"""

import cv2
import numpy as np

from pipeline.strokes import Material, Region, StrokeFrame
from pipeline.vectorize import absorb_small_regions, trace_regions

from .base import FrameInput, Maps, Style, StyleParams, lerp
from .clean import INK, ink_paths, masked_ink, uniform_strokes


def _flatten(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Edge-preserving blur so k-means sees flat colour areas, not texture."""
    out = rgb
    for _ in range(2):
        out = cv2.bilateralFilter(out, d=0, sigmaColor=28, sigmaSpace=4 * scale)
    return out


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)


def _area(inp: FrameInput, params: StyleParams) -> np.ndarray:
    if params.remove_background and inp.mask is not None:
        return inp.mask
    return np.ones(inp.rgb.shape[:2], bool)


def prepare(frames: list[FrameInput], params: StyleParams) -> np.ndarray:
    """Palette as (k, 3) Lab centres, from pixels of the sample frames."""
    k = int(round(lerp(4, 10, params.detail)))
    rng = np.random.default_rng(0)
    samples = []
    for f in frames:
        lab = _lab(_flatten(f.rgb, f.scale))[_area(f, params)]
        if len(lab):
            samples.append(lab[rng.choice(len(lab), min(len(lab), 40_000 // len(frames)), replace=False)])
    data = np.concatenate(samples).astype(np.float32) if samples else np.zeros((k, 3), np.float32)
    k = min(k, len(data))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, _, centres = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # Order light to dark so material names are stable and readable in Blender.
    return centres[np.argsort(-centres[:, 0])]


def _hex(lab: np.ndarray) -> str:
    rgb = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)[0, 0]
    r, g, b = np.clip(rgb * 255 + 0.5, 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def rasterize(inp: FrameInput, params: StyleParams, _: object) -> Maps:
    # Colour (Lab) is stabilised over time before it's quantised, so pixels near a
    # boundary between two palette colours don't flip back and forth.
    return {"lab": _lab(_flatten(inp.rgb, inp.scale)), "ink": masked_ink(inp, params)}


def vectorize(inp: FrameInput, maps: Maps, params: StyleParams, palette: np.ndarray) -> StrokeFrame:
    s = inp.scale
    w, h = inp.size
    lab = maps["lab"]
    # Nearest palette colour per pixel, one centre at a time to keep memory flat.
    best = np.full(lab.shape[:2], np.inf, np.float32)
    labels = np.zeros(lab.shape[:2], np.int32)
    for i, centre in enumerate(palette):
        d = ((lab - centre) ** 2).sum(-1)
        closer = d < best
        best[closer], labels[closer] = d[closer], i
    labels[~_area(inp, params)] = -1
    labels = absorb_small_regions(labels, int(lerp(700, 90, params.detail) * s * s))

    regions = [
        Region(f"fill_{lab_idx:02d}", outline, holes)
        for lab_idx, outline, holes, _ in trace_regions(
            labels, min_area=lerp(300, 40, params.detail) * s * s,
            sigma=lerp(0.8, 4.0, params.smoothing) * s, spacing=3 * s,
        )
    ]
    width = 1.8 * params.thickness * s
    return StrokeFrame(
        width=w, height=h,
        materials=[Material(f"fill_{i:02d}", _hex(c), fill=True) for i, c in enumerate(palette)]
        + [Material("ink", INK)],
        regions=regions,
        strokes=uniform_strokes(ink_paths(maps["ink"], s, params), width, "ink"),
    )


STYLE = Style(
    key="rotoscope",
    label="Rotoscope",
    description="Flat colour regions from a fixed palette inside the subject, clean outlines on top.",
    needs_neural_lines=True,
    prepare=prepare,
    rasterize=rasterize,
    vectorize=vectorize,
    background=lambda params: "#ffffff",
)
