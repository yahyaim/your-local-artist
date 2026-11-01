"""Shared types and helpers for the drawing styles."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from pipeline.strokes import StrokeFrame
from pipeline.vectorize import path_length, resample, skeleton_paths, smooth_path


class StyleParams(BaseModel):
    """User-facing look settings (the Settings panel). All styles accept all of them."""

    thickness: float = Field(1.0, ge=0.25, le=4.0, description="Line width multiplier")
    detail: float = Field(0.5, ge=0.0, le=1.0, description="0 = only strong lines, 1 = every faint edge")
    smoothing: float = Field(0.5, ge=0.0, le=1.0, description="Spatial (and, for video, temporal) smoothing")
    remove_background: bool = True
    paper_texture: bool = True  # pencil style only; preview/raster output only

    def cache_key(self) -> str:
        return self.model_dump_json()


@dataclass
class FrameInput:
    rgb: np.ndarray  # (H, W, 3) uint8
    mask: np.ndarray | None  # (H, W) bool subject mask, None if no subject
    neural_ink: np.ndarray | None = None  # (H, W) float32, set when the style needs it
    # How far the subject has moved since the first frame (x, y px), from optical flow.
    # Styles with screen-space patterns (pencil wobble, hatching) shift them by
    # this so the patterns travel with the subject instead of the subject sliding
    # through them. Always (0, 0) for single-frame previews.
    anchor: tuple[float, float] = (0.0, 0.0)

    @property
    def scale(self) -> float:
        """Size factor relative to the 720 px reference all pixel constants are tuned for."""
        return max(self.rgb.shape[:2]) / 720.0

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


Maps = dict[str, np.ndarray]


@dataclass(frozen=True)
class Style:
    """A drawing style, split in two so video generation can stabilise in between:

    rasterize -> float maps (line ink, shading, colour...) that the temporal
                 filter blends along the optical flow;
    vectorize -> strokes and regions traced from those (stabilised) maps.
    """

    key: str
    label: str
    description: str
    needs_neural_lines: bool
    # Called once per video (or preview) on a few sample frames, e.g. to fix a colour palette.
    prepare: Callable[[list[FrameInput], StyleParams], Any]
    rasterize: Callable[[FrameInput, StyleParams, Any], Maps]
    vectorize: Callable[[FrameInput, Maps, StyleParams, Any], StrokeFrame]
    background: Callable[[StyleParams], str]  # "#rrggbb" or "paper"

    def build(self, inp: FrameInput, params: StyleParams, state: Any) -> StrokeFrame:
        """Single frame, no temporal filtering (used for previews)."""
        return self.vectorize(inp, self.rasterize(inp, params, state), params, state)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def neural_input(rgb: np.ndarray, mask: np.ndarray | None, remove_background: bool) -> np.ndarray:
    """The image fed to the line network: with background removal, the subject on
    white, so the network draws the silhouette and nothing behind it."""
    if not remove_background or mask is None:
        return rgb
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)


def subject_region(inp: FrameInput, params: StyleParams, grow_px: float) -> np.ndarray | None:
    """Mask (grown slightly) that lines must stay inside, or None for the whole frame."""
    if not params.remove_background or inp.mask is None:
        return None
    r = max(1, int(round(grow_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(inp.mask.astype(np.uint8), kernel).astype(bool)


def centreline_paths(
    binary: np.ndarray, scale: float, smoothing: float, min_length: float, spacing: float = 2.5
) -> list[tuple[np.ndarray, bool]]:
    """Skeletonise, smooth and resample a binary line image into clean polylines."""
    out = []
    sigma_px = lerp(0.6, 5.0, smoothing) * scale
    for pts, closed in skeleton_paths(binary):
        if path_length(pts, closed) < min_length:
            continue
        smoothed = smooth_path(pts, closed, sigma_px)  # pixels are ~1 px apart, so sigma in points ≈ px
        out.append((resample(smoothed, spacing * scale, closed), closed))
    return out
