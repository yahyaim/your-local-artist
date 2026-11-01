"""Temporal stabilisation: the main defence against flicker.

Two complementary filters, both motion-compensated with backward optical flow
and both causal (frame t only looks at the already-smoothed frame t-1):

1. `RasterStabilizer`, before vectorising: each raster the style vectorises
   (line ink, pencil edges/shading, rotoscope colour, the subject mask) is
   blended with the previous frame's *smoothed* raster warped into place. A
   faint line therefore can't blink on and off between frames, and region
   colours can't flip. Where the flow is unreliable (occlusions, fast motion;
   see `flow.validity`) the current frame is used as is, so nothing smears.

2. `smooth_strokes`, after vectorising: every stroke point is matched to the
   nearest point of the previous smoothed frame (after following the flow
   back) and pulled towards it, which removes the sub-pixel wobble that
   re-tracing every frame from scratch would otherwise produce.

Both take a strength in [0, 1] derived from the user's Smoothing slider.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .flow import sample, warp
from .strokes import Region, Stroke, StrokeFrame


class RasterStabilizer:
    """Motion-compensated temporal filter for named float maps.

    Two modes:
    - Smooth fields (shading, colour, masks) use an exponential moving average
      along the flow.
    - Line-ink maps (keys in `gated`) use an evidence-gated max instead:
          out = max(cur, min(warped_prev, k * cur))
      A line seen last frame survives a frame where it's only faintly detected
      (no blinking), but nothing can appear where the current frame has no ink
      at all (no ghost trails), and current lines are never dimmed. Plain
      averaging would dim any line the flow misaligns by a pixel, which then
      falls below the threshold and vanishes, the opposite of what we want.
    """

    def __init__(self, strength: float, gated: frozenset[str] = frozenset({"ink"})) -> None:
        self.strength = strength
        self.gated = gated
        self.prev: dict[str, np.ndarray] = {}

    def __call__(self, maps: dict[str, np.ndarray], flow: np.ndarray | None,
                 valid: np.ndarray | None) -> dict[str, np.ndarray]:
        out = {}
        for key, cur in maps.items():
            prev = self.prev.get(key)
            if prev is None or flow is None or valid is None or self.strength <= 0 or prev.shape != cur.shape:
                out[key] = cur
                continue
            warped = warp(prev, flow)
            if key in self.gated:
                k = 1.0 + 2.0 * self.strength
                target = np.maximum(cur, np.minimum(warped, k * cur))
                out[key] = cur + valid * (target - cur)
            else:
                a = (self.strength * valid).astype(np.float32)
                if cur.ndim == 3:
                    a = a[..., None]
                out[key] = a * warped + (1 - a) * cur
        self.prev = out
        return out


def _index(points: list[np.ndarray], extra: list[np.ndarray]) -> tuple[cKDTree | None, np.ndarray, np.ndarray]:
    if not points:
        return None, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pts = np.concatenate(points)
    return cKDTree(pts), pts, np.concatenate(extra) if extra else np.zeros((len(pts), 2), np.float32)


def _pull(points: np.ndarray, closed: bool, tree: cKDTree | None, prev_pts: np.ndarray,
          flow: np.ndarray, valid: np.ndarray, strength: float, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """New positions for `points`, plus the matched previous index and blend weight per point."""
    n = len(points)
    if tree is None or n == 0:
        return points, np.full(n, -1), np.zeros(n, np.float32)
    back = sample(flow, points)  # where each point was in the previous frame
    dist, idx = tree.query(points + back, distance_upper_bound=radius)
    matched = np.isfinite(dist)
    weight = np.zeros(n, np.float32)
    weight[matched] = strength * sample(valid, points[matched]) * (1 - dist[matched] / radius)
    target = points.copy()
    # The matched (smoothed) previous point, carried forward along the flow.
    target[matched] = prev_pts[idx[matched]] - back[matched]
    delta = (target - points) * weight[:, None]
    if n >= 3:
        # Smooth the correction along the stroke so partial matches don't make kinks.
        delta = gaussian_filter1d(delta, 2.0, axis=0, mode="wrap" if closed else "nearest")
    return (points + delta).astype(np.float32), np.where(matched, idx, -1), weight


def smooth_strokes(prev: StrokeFrame, cur: StrokeFrame, flow: np.ndarray, valid: np.ndarray,
                   strength: float, radius: float) -> StrokeFrame:
    """Pull `cur`'s stroke points (and region outlines) towards the smoothed previous frame."""
    if strength <= 0:
        return cur

    # Indexes of the previous frame's points, built per material on first use.
    stroke_trees: dict[str, tuple] = {}
    strokes: list[Stroke] = []
    for s in cur.strokes:
        if s.material not in stroke_trees:
            same = [p for p in prev.strokes if p.material == s.material]
            stroke_trees[s.material] = _index(
                [p.points for p in same], [np.column_stack([p.widths, p.opacities]) for p in same]
            )
        tree, prev_pts, prev_wo = stroke_trees[s.material]
        pts, idx, weight = _pull(s.points, s.closed, tree, prev_pts, flow, valid, strength, radius)
        widths, opacities = s.widths.copy(), s.opacities.copy()
        m = idx >= 0
        if m.any():
            # Width and opacity follow too, so pencil pressure doesn't pulse.
            widths[m] += weight[m] * (prev_wo[idx[m], 0] - widths[m])
            opacities[m] += weight[m] * (prev_wo[idx[m], 1] - opacities[m])
        strokes.append(Stroke(s.material, pts, widths, opacities, s.closed))

    regions: list[Region] = []
    trees: dict[str, tuple] = {}
    for r in cur.regions:
        if r.material not in trees:
            same = [p for q in prev.regions if q.material == r.material for p in [q.outline, *q.holes]]
            trees[r.material] = _index(same, [])[:2]
        tree, prev_pts = trees[r.material]
        outline = _pull(r.outline, True, tree, prev_pts, flow, valid, strength, radius)[0]
        holes = [_pull(h, True, tree, prev_pts, flow, valid, strength, radius)[0] for h in r.holes]
        regions.append(Region(r.material, outline, holes))

    return StrokeFrame(cur.width, cur.height, cur.materials, regions, strokes)
