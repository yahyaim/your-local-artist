"""Optical flow (RAFT, torchvision) and flow-based warping.

Only backward flow is computed: for frame t, `flow[y, x]` points from pixel
(x, y) in frame t to where that content was in frame t-1. That's all the
causal temporal filters need: warp the previous result into the current frame
(`warp`), or look up where a current point came from (`sample`).

Flow is computed and cached at a reduced size (long side 512, a multiple of 8
as RAFT requires) and scaled up on load; it's smooth enough that this costs
little accuracy and keeps the cache to a few hundred KB per frame.
"""

import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from . import weights
from .device import device_info, torch_device
from .subject import gpu_lock

FLOW_LONG_SIDE = 512

_models: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


def raft_variant(low_memory: bool) -> str:
    return "raft_small" if low_memory or device_info().is_slow else "raft_large"


def _model(variant: str) -> torch.nn.Module:
    with _lock:
        if variant not in _models:
            from torchvision.models.optical_flow import raft_large, raft_small

            model = (raft_large if variant == "raft_large" else raft_small)(weights=None)
            model.load_state_dict(torch.load(weights.ensure(variant), map_location="cpu", weights_only=True))
            _models[variant] = model.eval().to(torch_device())
        return _models[variant]


def release_models() -> None:
    with _lock:
        _models.clear()


def flow_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, FLOW_LONG_SIDE / max(width, height))
    return max(8, int(round(width * scale / 8)) * 8), max(8, int(round(height * scale / 8)) * 8)


def _to_tensor(rgbs: list[np.ndarray]) -> torch.Tensor:
    x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
    return (x / 127.5 - 1.0).to(torch_device())  # RAFT expects [-1, 1]


@torch.inference_mode()
def compute_backward_flows(
    frames: list[Path], out_dir: Path, variant: str, batch: int,
    on_progress: Callable[[int], None], check_cancel: Callable[[], None],
) -> None:
    """Write out_dir/<t>.npy (float16, flow-size) for t = 1..n-1: flow from frame t to t-1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames[0]))
    fw, fh = flow_size(first.shape[1], first.shape[0])

    def load(i: int) -> np.ndarray:
        return cv2.resize(cv2.imread(str(frames[i]))[..., ::-1], (fw, fh), interpolation=cv2.INTER_AREA)

    model = _model(variant)
    prev = load(0)
    on_progress(1)
    t = 1
    while t < len(frames):
        check_cancel()
        idx = list(range(t, min(t + batch, len(frames))))
        cur = [load(i) for i in idx]
        befores = [prev] + cur[:-1]
        with gpu_lock:
            flows = model(_to_tensor(cur), _to_tensor(befores))[-1]  # final refinement
        for i, f in zip(idx, flows.permute(0, 2, 3, 1).cpu().numpy()):
            np.save(out_dir / f"{i:05d}.npy", f.astype(np.float16))
        prev = cur[-1]
        t = idx[-1] + 1
        on_progress(t)


def load_flow(out_dir: Path, t: int, size: tuple[int, int]) -> np.ndarray:
    """Backward flow for frame t, scaled to `size` (w, h) in output pixels."""
    w, h = size
    f = np.load(out_dir / f"{t:05d}.npy").astype(np.float32)
    fh, fw = f.shape[:2]
    up = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
    up[..., 0] *= w / fw
    up[..., 1] *= h / fh
    return up


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Bring frame t-1 content into frame t: out[y, x] = image[y + fy, x + fx]."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def validity(cur_rgb: np.ndarray, prev_rgb: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Per-pixel confidence in [0, 1] that the flow is right, from how well the warped
    previous frame matches the current one. Low at occlusions, motion blur and
    where content leaves the frame, so history isn't smeared there."""
    cur = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    prev = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    err = cv2.GaussianBlur(np.abs(cur - warp(prev, flow)), (0, 0), 2.0)
    conf = np.clip(1.0 - (err - 0.04) / 0.08, 0.0, 1.0)
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    inside = (gx + flow[..., 0] >= 0) & (gx + flow[..., 0] <= w - 1) & (gy + flow[..., 1] >= 0) & (gy + flow[..., 1] <= h - 1)
    return conf * inside


def sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear lookup of an (H, W[, C]) field at float (x, y) points."""
    h, w = field.shape[:2]
    x = np.clip(points[:, 0], 0, w - 1.001)
    y = np.clip(points[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    if field.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    top = field[y0, x0] * (1 - fx) + field[y0, x0 + 1] * fx
    bottom = field[y0 + 1, x0] * (1 - fx) + field[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy
