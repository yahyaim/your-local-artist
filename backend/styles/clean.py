"""Clean cartoon outlines: SAM 2 subject + neural line art, uniform smooth strokes."""

import numpy as np
from skimage.filters import apply_hysteresis_threshold

from pipeline.strokes import Material, Stroke, StrokeFrame
from pipeline.vectorize import remove_small_components

from .base import FrameInput, Maps, Style, StyleParams, centreline_paths, lerp, subject_region

INK = "#17262b"


def masked_ink(inp: FrameInput, params: StyleParams) -> np.ndarray:
    """Neural line ink, limited to the subject when background is removed."""
    assert inp.neural_ink is not None
    region = subject_region(inp, params, 3 * inp.scale)
    return inp.neural_ink * region if region is not None else inp.neural_ink


def ink_paths(ink: np.ndarray, scale: float, params: StyleParams) -> list[tuple[np.ndarray, bool]]:
    # Hysteresis: faint line pixels survive when connected to strong ones, so
    # lines stay continuous instead of breaking into dashes.
    binary = apply_hysteresis_threshold(ink, lerp(0.3, 0.12, params.detail), lerp(0.6, 0.35, params.detail))
    binary = remove_small_components(binary, int(12 * scale * scale))
    return centreline_paths(binary, scale, params.smoothing, min_length=lerp(18, 6, params.detail) * scale)


def uniform_strokes(paths: list[tuple[np.ndarray, bool]], width: float, material: str) -> list[Stroke]:
    return [
        Stroke(material, pts, np.full(len(pts), width, np.float32), np.ones(len(pts), np.float32), closed)
        for pts, closed in paths
        if len(pts) >= 2
    ]


def rasterize(inp: FrameInput, params: StyleParams, _: object) -> Maps:
    return {"ink": masked_ink(inp, params)}


def vectorize(inp: FrameInput, maps: Maps, params: StyleParams, _: object) -> StrokeFrame:
    w, h = inp.size
    width = 2.4 * params.thickness * inp.scale
    return StrokeFrame(
        width=w, height=h,
        materials=[Material("ink", INK)],
        strokes=uniform_strokes(ink_paths(maps["ink"], inp.scale, params), width, "ink"),
    )


STYLE = Style(
    key="clean",
    label="Clean cartoon outlines",
    description="Subject isolated with SAM 2, neural line art, smooth uniform strokes, no background.",
    needs_neural_lines=True,
    prepare=lambda frames, params: None,
    rasterize=rasterize,
    vectorize=vectorize,
    background=lambda params: "#ffffff",
)
