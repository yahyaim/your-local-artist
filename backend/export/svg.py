"""Stroke JSON -> SVG (one file per frame), and the zip of all frames.

Mapping:
- Regions: one <path> per region, outline + holes, fill-rule="evenodd".
- Uniform-width strokes: a stroked <path> with round caps/joins, smoothed with
  quadratic curves through segment midpoints (the same curve the preview draws).
- Variable-width strokes (pencil): SVG has no per-point stroke width, so each
  becomes a filled outline polygon (left edge, round end cap, right edge, round
  start cap). Per-point opacity is approximated by the stroke's mean opacity.
"""

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from pipeline.strokes import Region, Stroke, StrokeFrame


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _poly(points: np.ndarray, close: bool = True) -> str:
    d = "M" + " L".join(f"{_f(x)} {_f(y)}" for x, y in points)
    return d + " Z" if close else d


def _smooth_path(points: np.ndarray, closed: bool) -> str:
    if len(points) < 3:
        return _poly(points, closed)
    parts = [f"M{_f(points[0, 0])} {_f(points[0, 1])}"]
    for i in range(1, len(points) - 1):
        mx, my = (points[i] + points[i + 1]) / 2
        parts.append(f"Q{_f(points[i, 0])} {_f(points[i, 1])} {_f(mx)} {_f(my)}")
    parts.append(f"L{_f(points[-1, 0])} {_f(points[-1, 1])}")
    return " ".join(parts) + (" Z" if closed else "")


def _outline(stroke: Stroke) -> str:
    """Filled outline of a variable-width stroke, with round caps."""
    p = stroke.points.astype(np.float64)
    r = stroke.widths.astype(np.float64) / 2
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.column_stack([-t[:, 1], t[:, 0]])
    left, right = p + n * r[:, None], p - n * r[:, None]
    if stroke.closed:
        # A closed band: outer ring plus inner ring, filled with evenodd.
        return _poly(left) + " " + _poly(right[::-1])

    def cap(radius: float, to: np.ndarray) -> str:
        return f"A{_f(radius)} {_f(radius)} 0 0 1 {_f(to[0])} {_f(to[1])}"

    return (
        _poly(left, close=False)
        + " " + cap(r[-1], right[-1])
        + " L" + " L".join(f"{_f(x)} {_f(y)}" for x, y in right[::-1])
        + " " + cap(r[0], left[0]) + " Z"
    )


def frame_to_svg(frame: StrokeFrame) -> str:
    colors = {m.name: m.color for m in frame.materials}
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{frame.width}" height="{frame.height}" '
        f'viewBox="0 0 {frame.width} {frame.height}">',
    ]
    if frame.regions:
        out.append('<g id="fills">')
        for region in frame.regions:
            out.append(_region(region, colors[region.material]))
        out.append("</g>")
    out.append('<g id="strokes">')
    for stroke in frame.strokes:
        if len(stroke.points) < 2:
            continue
        color, opacity = colors[stroke.material], float(stroke.opacities.mean())
        cls = f'class="{escape(stroke.material)}"'
        if np.ptp(stroke.widths) < 1e-3:
            out.append(
                f'<path {cls} d="{_smooth_path(stroke.points, stroke.closed)}" fill="none" stroke="{color}" '
                f'stroke-width="{_f(float(stroke.widths[0]))}" stroke-linecap="round" stroke-linejoin="round"'
                + (f' stroke-opacity="{opacity:.3f}"' if opacity < 0.999 else "") + "/>"
            )
        else:
            rule = ' fill-rule="evenodd"' if stroke.closed else ""
            out.append(
                f'<path {cls} d="{_outline(stroke)}" fill="{color}"{rule}'
                + (f' fill-opacity="{opacity:.3f}"' if opacity < 0.999 else "") + "/>"
            )
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def _region(region: Region, color: str) -> str:
    d = " ".join(_poly(ring) for ring in [region.outline, *region.holes] if len(ring) >= 3)
    return f'<path class="{escape(region.material)}" d="{d}" fill="{color}" fill-rule="evenodd"/>'


def build_svg_zip(strokes_dir: Path, out: Path) -> Path:
    """Zip of frames/frame_00000.svg ... Written to a temp name first so a failed
    build never leaves a truncated zip that looks finished."""
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(strokes_dir.glob("*.json")):
            frame = StrokeFrame.from_json(json.loads(path.read_text()))
            zf.writestr(f"frames/frame_{path.stem}.svg", frame_to_svg(frame))
    tmp.rename(out)
    return out
