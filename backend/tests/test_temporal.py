import numpy as np

from pipeline.flow import sample, warp
from pipeline.strokes import Material, Stroke, StrokeFrame
from pipeline.temporal import RasterStabilizer, smooth_strokes

H, W = 60, 80
ZERO_FLOW = np.zeros((H, W, 2), np.float32)
ALL_VALID = np.ones((H, W), np.float32)


def test_warp_follows_backward_flow() -> None:
    img = np.zeros((H, W), np.float32)
    img[20, 30] = 1.0
    flow = np.zeros((H, W, 2), np.float32)
    flow[..., 0] = -5  # content moved 5 px right: frame t at x came from x-5 in frame t-1
    out = warp(img, flow)
    assert out[20, 35] == 1.0 and out[20, 30] == 0.0


def test_sample_is_bilinear() -> None:
    field = np.zeros((4, 4), np.float32)
    field[1, 2] = 1.0
    assert np.isclose(sample(field, np.array([[1.5, 1.0]], np.float32))[0], 0.5)


def test_gated_ink_never_dims_current_and_never_ghosts() -> None:
    stab = RasterStabilizer(1.0)
    line = np.zeros((H, W), np.float32)
    line[30, 10:70] = 0.9
    stab({"ink": line}, None, None)

    # Next frame: the line is only faintly detected; it should be lifted back up.
    faint = line * 0.3
    out = stab({"ink": faint}, ZERO_FLOW, ALL_VALID)["ink"]
    assert (out >= faint).all()
    assert out[30, 40] > 0.5

    # Next frame: no ink at all; the old line must not survive as a ghost.
    out = stab({"ink": np.zeros((H, W), np.float32)}, ZERO_FLOW, ALL_VALID)["ink"]
    assert out.max() == 0.0


def test_ema_respects_validity() -> None:
    stab = RasterStabilizer(0.5)
    stab({"lum": np.ones((H, W), np.float32)}, None, None)
    valid = np.zeros((H, W), np.float32)
    valid[:, : W // 2] = 1.0
    out = stab({"lum": np.zeros((H, W), np.float32)}, ZERO_FLOW, valid)["lum"]
    assert np.isclose(out[0, 0], 0.5)  # blended where flow is trusted
    assert out[0, -1] == 0.0  # current frame only where it isn't


def _frame(y_offsets: np.ndarray) -> StrokeFrame:
    x = np.linspace(10, 70, 25, dtype=np.float32)
    pts = np.column_stack([x, 30 + y_offsets]).astype(np.float32)
    return StrokeFrame(W, H, [Material("ink", "#000000")],
                       strokes=[Stroke("ink", pts, np.full(25, 2, np.float32), np.ones(25, np.float32))])


def test_smooth_strokes_damps_jitter_of_a_static_line() -> None:
    # Re-tracing jitter is smooth along the line (paths are spatially smoothed):
    # each frame the line shifts and tilts by a random sub-pixel amount.
    rng = np.random.default_rng(0)
    u = np.linspace(-1, 1, 25)
    raw = [_frame(rng.normal(0, 0.6) + rng.normal(0, 0.4) * u) for _ in range(30)]
    smoothed = [raw[0]]
    for cur in raw[1:]:
        smoothed.append(smooth_strokes(smoothed[-1], cur, ZERO_FLOW, ALL_VALID, 0.9, radius=3.0))

    def jitter(frames: list[StrokeFrame]) -> float:
        ys = np.stack([f.strokes[0].points[:, 1] for f in frames[10:]])
        return float(np.abs(np.diff(ys, axis=0)).mean())

    assert jitter(smoothed) < jitter(raw) / 2


def test_smooth_strokes_follows_motion_instead_of_lagging() -> None:
    # A line moving down 2 px per frame, with flow that says so, is not held back.
    flow = np.zeros((H, W, 2), np.float32)
    flow[..., 1] = -2  # content came from 2 px above
    prev = _frame(np.zeros(25))
    cur = _frame(np.full(25, 2.0))
    out = smooth_strokes(prev, cur, flow, ALL_VALID, 0.9, radius=3.0)
    assert np.allclose(out.strokes[0].points[:, 1], 32.0, atol=0.05)
