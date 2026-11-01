import cv2
import numpy as np
import pytest

from pipeline.render import render
from pipeline.strokes import Material, Region, Stroke, StrokeFrame
from pipeline.vectorize import (
    absorb_small_regions, path_length, resample, skeleton_paths, smooth_path, trace_regions,
)


def _canvas(h: int = 120, w: int = 200) -> np.ndarray:
    return np.zeros((h, w), np.uint8)


def test_straight_line_traces_to_one_open_path() -> None:
    img = _canvas()
    cv2.line(img, (20, 60), (180, 60), 255, 3)
    paths = skeleton_paths(img > 0)
    assert len(paths) == 1
    pts, closed = paths[0]
    assert not closed
    xs = sorted([pts[0][0], pts[-1][0]])
    assert xs[0] == pytest.approx(20, abs=3) and xs[1] == pytest.approx(180, abs=3)
    assert np.allclose(pts[:, 1], 60, atol=1.5)


def test_cross_splits_at_junction_and_meets_there() -> None:
    img = _canvas()
    cv2.line(img, (20, 60), (180, 60), 255, 3)
    cv2.line(img, (100, 10), (100, 110), 255, 3)
    paths = skeleton_paths(img > 0)
    assert len(paths) == 4
    # Every arm ends at (about) the junction centre.
    for pts, _ in paths:
        ends = [pts[0], pts[-1]]
        assert min(np.hypot(e[0] - 100, e[1] - 60) for e in ends) < 2.5


def test_circle_traces_to_closed_path() -> None:
    img = _canvas()
    cv2.circle(img, (100, 60), 40, 255, 2)
    paths = skeleton_paths(img > 0)
    assert len(paths) == 1
    pts, closed = paths[0]
    assert closed
    assert path_length(pts, closed) == pytest.approx(2 * np.pi * 40, rel=0.1)


def test_resample_gives_uniform_spacing_and_keeps_ends() -> None:
    pts = np.array([[0, 0], [10, 0], [10, 10]], np.float32)
    out = resample(pts, 2.0)
    gaps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(gaps, gaps.mean(), atol=0.6)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])


def test_smooth_reduces_noise_but_pins_endpoints() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 101)
    noisy = np.column_stack([x, rng.normal(0, 2, x.size)]).astype(np.float32)
    out = smooth_path(noisy, False, sigma=4)
    assert np.abs(out[1:-1, 1]).std() < np.abs(noisy[1:-1, 1]).std() / 2
    assert np.allclose(out[0], noisy[0]) and np.allclose(out[-1], noisy[-1])


def test_trace_regions_finds_holes_and_orders_by_area() -> None:
    labels = np.full((100, 100), -1, np.int32)
    labels[10:90, 10:90] = 0
    labels[30:70, 30:70] = -1  # hole showing the background
    labels[45:55, 45:55] = 1  # small region inside the hole
    regions = trace_regions(labels, min_area=10, sigma=0, spacing=2)
    assert [r[0] for r in regions] == [0, 1]
    assert len(regions[0][2]) == 1  # the ring has one hole
    assert regions[0][3] > regions[1][3]


def test_absorb_small_regions_merges_specks_into_neighbours() -> None:
    labels = np.zeros((50, 50), np.int32)
    labels[20:22, 20:22] = 1  # 4-pixel speck
    labels[:, 40:] = -1  # outside the subject stays outside
    out = absorb_small_regions(labels, min_area=10)
    assert (out[20:22, 20:22] == 0).all()
    assert (out[:, 40:] == -1).all()


def test_stroke_frame_json_round_trip() -> None:
    frame = StrokeFrame(
        width=64, height=32,
        materials=[Material("ink", "#000000"), Material("fill_00", "#ff0000", fill=True)],
        regions=[Region("fill_00", np.array([[1, 1], [10, 1], [10, 10]], np.float32), [])],
        strokes=[Stroke("ink", np.array([[0, 0], [5.123, 6]], np.float32),
                        np.array([2, 3], np.float32), np.array([1, 0.5], np.float32), False)],
    )
    back = StrokeFrame.from_json(frame.to_json())
    assert back.width == 64 and back.material("fill_00").fill
    assert np.allclose(back.strokes[0].points, [[0, 0], [5.12, 6]])
    assert np.allclose(back.strokes[0].opacities, [1, 0.5])
    assert len(back.regions[0].outline) == 3


def test_render_draws_strokes_and_region_holes() -> None:
    frame = StrokeFrame(
        width=100, height=100,
        materials=[Material("ink", "#000000"), Material("fill_00", "#ff0000", fill=True)],
        regions=[Region(
            "fill_00",
            np.array([[50, 50], [95, 50], [95, 95], [50, 95]], np.float32),
            [np.array([[65, 65], [80, 65], [80, 80], [65, 80]], np.float32)],
        )],
        strokes=[Stroke("ink", np.array([[10, 20], [40, 20]], np.float32),
                        np.array([4, 4], np.float32), np.array([1, 1], np.float32))],
    )
    img = render(frame)
    assert img[20, 25, 3] == 255 and img[20, 25, :3].max() == 0  # opaque black stroke
    assert tuple(img[55, 55]) == (255, 0, 0, 255)  # filled region
    assert img[72, 72, 3] == 0  # the hole stays transparent
    assert img[5, 5, 3] == 0
