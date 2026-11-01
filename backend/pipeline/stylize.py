"""Run a style on frames: line extraction -> vector strokes -> rendered image."""

import numpy as np

from styles import STYLES, FrameInput, StyleParams
from styles.base import neural_input

from . import lines
from .render import composite, paper_texture, render
from .strokes import StrokeFrame
from .subject import gpu_lock


def neural_ink(rgb: np.ndarray, mask: np.ndarray | None, params: StyleParams) -> np.ndarray:
    with gpu_lock:
        return lines.neural_lines(neural_input(rgb, mask, params.remove_background)[None])[0]


def background_image(style_key: str, params: StyleParams, width: int, height: int) -> np.ndarray:
    bg = STYLES[style_key].background(params)
    if bg == "paper":
        return paper_texture(width, height)
    h = bg.lstrip("#")
    return np.full((height, width, 3), [int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def stylize_frame(
    style_key: str, rgb: np.ndarray, mask: np.ndarray | None, params: StyleParams,
    ink: np.ndarray | None = None,
) -> tuple[StrokeFrame, np.ndarray]:
    """Single-frame preview: returns the vector frame and its rendering (RGB on the style's background).

    Video generation (M3) runs the same steps, but prepares the style once per
    video and batches the neural line extraction.
    """
    style = STYLES[style_key]
    inp = FrameInput(rgb=rgb, mask=mask)
    if style.needs_neural_lines:
        inp.neural_ink = ink if ink is not None else neural_ink(rgb, mask, params)
    state = style.prepare([inp], params)
    frame = style.build(inp, params, state)
    h, w = rgb.shape[:2]
    image = composite(render(frame), background_image(style_key, params, w, h))
    return frame, image
