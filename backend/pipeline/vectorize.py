"""Raster -> vector: skeleton centreline tracing and region outline tracing."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize

_NEIGHBOURS_4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_NEIGHBOURS_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_COUNT_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)


def remove_small_components(binary: np.ndarray, min_pixels: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
    return keep[labels]


def _walk(pixels: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    """Follow a 1-pixel-wide chain from `start`, consuming pixels from the set.

    4-neighbours are preferred over diagonals so staircase corners don't make
    the walk skip a pixel and strand it.
    """
    path = [start]
    pixels.discard(start)
    y, x = start
    while True:
        for dy, dx in _NEIGHBOURS_4 + _NEIGHBOURS_DIAG:
            nxt = (y + dy, x + dx)
            if nxt in pixels:
                pixels.discard(nxt)
                path.append(nxt)
                y, x = nxt
                break
        else:
            return path


def _degree(p: tuple[int, int], pixels: set[tuple[int, int]]) -> int:
    y, x = p
    return sum((y + dy, x + dx) in pixels for dy, dx in _NEIGHBOURS_4 + _NEIGHBOURS_DIAG)


def skeleton_paths(binary: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Trace the centrelines of a binary line mask.

    Returns `(points, closed)` pairs with points as float32 (N, 2) in (x, y).
    The skeleton is cut at junction pixels (3+ neighbours) into simple chains;
    each chain end that touches a junction is extended to the junction's
    centroid so strokes still meet there.
    """
    skel = skeletonize(binary.astype(bool)).astype(np.uint8)
    degree = cv2.filter2D(skel.astype(np.float32), -1, _COUNT_KERNEL, borderType=cv2.BORDER_CONSTANT) * skel
    junction = (degree >= 3).astype(np.uint8)
    _, junction_labels, _, junction_centroids = cv2.connectedComponentsWithStats(junction, connectivity=8)

    body = (skel == 1) & (junction == 0)
    n, labels = cv2.connectedComponents(body.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(body)
    comp = labels[ys, xs]
    order = np.argsort(comp, kind="stable")
    bounds = np.searchsorted(comp[order], np.arange(1, n + 1))
    h, w = binary.shape

    def junction_at(p: tuple[int, int]) -> int:
        y, x = p
        y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, h), max(x - 1, 0), min(x + 2, w)
        found = junction_labels[y0:y1, x0:x1]
        return int(found.max())

    paths: list[tuple[np.ndarray, bool]] = []
    for i in range(n - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        pixels = set(zip(ys[idx].tolist(), xs[idx].tolist()))
        while pixels:
            ends = [p for p in pixels if _degree(p, pixels) <= 1]
            loop = not ends
            chain = _walk(pixels, ends[0] if ends else next(iter(pixels)))
            pts = [(float(x), float(y)) for y, x in chain]
            closed = loop and len(chain) > 2
            if not closed:
                for at_end in (False, True):
                    j = junction_at(chain[-1] if at_end else chain[0])
                    if j:
                        c = (float(junction_centroids[j][0]), float(junction_centroids[j][1]))
                        pts = pts + [c] if at_end else [c] + pts
            if len(pts) >= 2:
                paths.append((np.asarray(pts, np.float32), closed))
    return paths


def path_length(points: np.ndarray, closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.vstack([points, points[:1]]) if closed else points
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def smooth_path(points: np.ndarray, closed: bool, sigma: float) -> np.ndarray:
    """Gaussian-smooth a polyline along its length (sigma in points).

    Open paths keep their exact endpoints so strokes still meet at junctions.
    """
    if sigma <= 0 or len(points) < 3:
        return points
    out = gaussian_filter1d(points.astype(np.float32), sigma, axis=0, mode="wrap" if closed else "nearest")
    if not closed:
        out[0], out[-1] = points[0], points[-1]
    return out


def resample(points: np.ndarray, spacing: float, closed: bool = False) -> np.ndarray:
    """Resample a polyline at (approximately) uniform arc-length spacing."""
    pts = np.vstack([points, points[:1]]) if closed else points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return points[:1].copy()
    n = max(int(round(total / spacing)), 1)
    t = np.linspace(0.0, total, n + 1)
    out = np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]).astype(np.float32)
    return out[:-1] if closed and len(out) > 3 else out


def sample_map(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a 2D map at (x, y) points."""
    h, w = image.shape[:2]
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)
    return image[ys, xs]


def absorb_small_regions(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Relabel connected components smaller than `min_area` to the nearest large region.

    `labels` is int32 with -1 meaning "no region" (kept as is).
    """
    small = np.zeros(labels.shape, bool)
    for lab in np.unique(labels[labels >= 0]):
        n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == lab).astype(np.uint8), connectivity=8)
        tiny = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_area)
        tiny = tiny[tiny > 0]
        if len(tiny):
            small |= np.isin(comp, tiny)
    known = (labels >= 0) & ~small
    if not small.any() or not known.any():
        return labels
    # For every pixel, the index of the nearest "known" pixel.
    _, (iy, ix) = distance_transform_edt(~known, return_indices=True)
    out = labels.copy()
    out[small] = labels[iy[small], ix[small]]
    return out


def trace_regions(
    labels: np.ndarray, min_area: float, sigma: float, spacing: float
) -> list[tuple[int, np.ndarray, list[np.ndarray], float]]:
    """Trace each label's connected areas to (label, outline, holes, area), largest first.

    Each label mask is dilated by one pixel before tracing so neighbouring
    fills overlap slightly instead of leaving hairline gaps between them.
    """
    kernel = np.ones((3, 3), np.uint8)
    out = []
    for lab in np.unique(labels[labels >= 0]):
        mask = cv2.dilate((labels == lab).astype(np.uint8), kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue  # a hole; collected with its parent below
            area = cv2.contourArea(contour)
            if area < min_area or len(contour) < 3:
                continue

            def clean(c: np.ndarray) -> np.ndarray:
                pts = c[:, 0, :].astype(np.float32)
                return resample(smooth_path(pts, True, sigma), spacing, closed=True)

            holes = []
            child = hierarchy[i][2]
            while child != -1:
                if cv2.contourArea(contours[child]) >= min_area and len(contours[child]) >= 3:
                    holes.append(clean(contours[child]))
                child = hierarchy[child][0]
            out.append((int(lab), clean(contour), holes, float(area)))
    out.sort(key=lambda r: -r[3])
    return out
