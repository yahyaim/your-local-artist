"""Rasterise a StrokeFrame with Skia, so previews match the vector data exactly."""

from functools import lru_cache

import cv2
import numpy as np
import skia

from .strokes import StrokeFrame


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=4)
def paper_texture(width: int, height: int) -> np.ndarray:
    """Procedural paper (RGB uint8): soft blotches plus fine grain, seeded so it doesn't flicker."""
    rng = np.random.default_rng(7)
    blotch = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), 18)
    grain = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), 0.7)
    fibre = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=0.6)
    tone = 1.0 + 0.9 * blotch / (np.abs(blotch).max() + 1e-6) * 0.03 + grain * 0.025 + fibre * 0.02
    base = np.array([245, 243, 238], np.float32)  # warm off-white
    return np.clip(base * tone[..., None], 0, 255).astype(np.uint8)


def _polygon(path: skia.Path, pts: np.ndarray) -> None:
    path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
    for x, y in pts[1:]:
        path.lineTo(float(x), float(y))
    path.close()


def render(frame: StrokeFrame, background: np.ndarray | str | None = None) -> np.ndarray:
    """Return an RGBA uint8 image (straight alpha).

    `background` is an RGB image, a "#rrggbb" colour, or None for transparent.
    """
    out = np.zeros((frame.height, frame.width, 4), np.uint8)
    surface = skia.Surface(out, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    canvas = surface.getCanvas()
    if isinstance(background, str):
        r, g, b = _rgb(background)
        canvas.clear(skia.ColorSetARGB(255, r, g, b))
    elif background is not None:
        rgba = np.dstack([background, np.full(background.shape[:2], 255, np.uint8)])
        canvas.drawImage(skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType), 0, 0)

    materials = {m.name: m for m in frame.materials}

    for region in frame.regions:
        r, g, b = _rgb(materials[region.material].color)
        path = skia.Path()
        path.setFillType(skia.PathFillType.kEvenOdd)
        _polygon(path, region.outline)
        for hole in region.holes:
            _polygon(path, hole)
        canvas.drawPath(path, skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(255, r, g, b)))

    for stroke in frame.strokes:
        pts, widths, alphas = stroke.points, stroke.widths, stroke.opacities
        if len(pts) < 2:
            continue
        r, g, b = _rgb(materials[stroke.material].color)
        uniform = np.ptp(widths) < 1e-3 and np.ptp(alphas) < 1e-3
        if uniform:
            # Fast path: one smooth path (quadratic through midpoints) with one paint.
            path = skia.Path()
            path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
            for i in range(1, len(pts) - 1):
                mx, my = (pts[i] + pts[i + 1]) / 2
                path.quadTo(float(pts[i, 0]), float(pts[i, 1]), float(mx), float(my))
            path.lineTo(float(pts[-1, 0]), float(pts[-1, 1]))
            if stroke.closed:
                path.close()
            canvas.drawPath(path, skia.Paint(
                AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=float(widths[0]),
                StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join,
                Color=skia.ColorSetARGB(int(alphas[0] * 255), r, g, b),
            ))
            continue

        # Variable width/opacity: draw segments into an isolated layer with
        # kSrc blending, so overlapping round caps replace rather than stack
        # (no dark beads at joints); the layer then composites normally.
        pad = float(widths.max()) + 2
        bounds = skia.Rect.MakeLTRB(
            float(pts[:, 0].min()) - pad, float(pts[:, 1].min()) - pad,
            float(pts[:, 0].max()) + pad, float(pts[:, 1].max()) + pad,
        )
        canvas.saveLayer(bounds)
        seg = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap,
                         BlendMode=skia.BlendMode.kSrc)
        n = len(pts) if stroke.closed else len(pts) - 1
        for i in range(n):
            j = (i + 1) % len(pts)
            seg.setStrokeWidth(float(widths[i] + widths[j]) / 2)
            seg.setColor(skia.ColorSetARGB(int((alphas[i] + alphas[j]) * 127.5), r, g, b))
            canvas.drawLine(float(pts[i, 0]), float(pts[i, 1]), float(pts[j, 0]), float(pts[j, 1]), seg)
        canvas.restore()

    surface.flushAndSubmit()
    # Skia wrote premultiplied pixels into `out`; un-premultiply for PNG/compositing.
    alpha = out[..., 3:4].astype(np.float32)
    rgb = np.where(alpha > 0, out[..., :3] * (255.0 / np.maximum(alpha, 1)), 0)
    return np.dstack([np.clip(rgb + 0.5, 0, 255).astype(np.uint8), out[..., 3]])


def composite(rgba: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Alpha-composite a straight-alpha RGBA image over an RGB background."""
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + background * (1 - a) + 0.5).astype(np.uint8)
