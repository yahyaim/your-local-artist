"""Raster line extraction: neural (Informative Drawings) and classical (XDoG).

Both return a float32 "ink" map in [0, 1] (1 = full line) at input resolution.
"""

import threading

import cv2
import numpy as np
import torch
from torch import nn

from . import weights
from .device import torch_device

# --- Informative Drawings generator ------------------------------------------
# Architecture from Chan et al., "Learning to generate line drawings that convey
# geometry and semantics" (CVPR 2022), matching the released checkpoints.


class _ResidualBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(features, features, 3), nn.InstanceNorm2d(features), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(features, features, 3), nn.InstanceNorm2d(features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class InformativeDrawingsGenerator(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 1, n_residual_blocks: int = 3) -> None:
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)
        )
        down: list[nn.Module] = []
        features = 64
        for _ in range(2):
            down += [nn.Conv2d(features, features * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(features * 2), nn.ReLU(inplace=True)]
            features *= 2
        self.model1 = nn.Sequential(*down)
        self.model2 = nn.Sequential(*[_ResidualBlock(features) for _ in range(n_residual_blocks)])
        up: list[nn.Module] = []
        for _ in range(2):
            up += [
                nn.ConvTranspose2d(features, features // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(features // 2), nn.ReLU(inplace=True),
            ]
            features //= 2
        self.model3 = nn.Sequential(*up)
        self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


_model: InformativeDrawingsGenerator | None = None
_model_lock = threading.Lock()


def _line_model() -> InformativeDrawingsGenerator:
    global _model
    with _model_lock:
        if _model is None:
            model = InformativeDrawingsGenerator()
            model.load_state_dict(torch.load(weights.ensure("line_contour"), map_location="cpu", weights_only=True))
            _model = model.eval().to(torch_device())
        return _model


def release_line_model() -> None:
    global _model
    with _model_lock:
        _model = None


@torch.inference_mode()
def neural_lines(rgb_batch: np.ndarray) -> np.ndarray:
    """Line ink for a batch of RGB uint8 frames (B, H, W, 3) -> (B, H, W) float32."""
    model = _line_model()
    b, h, w, _ = rgb_batch.shape
    # Two stride-2 downsamplings: pad to a multiple of 4 with edge pixels.
    ph, pw = (-h) % 4, (-w) % 4
    x = torch.from_numpy(rgb_batch).to(torch_device()).permute(0, 3, 1, 2).float().div_(255)
    if ph or pw:
        x = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
    out = model(x)[:, 0, :h, :w]
    # The network draws dark lines on white; flip to ink.
    return (1.0 - out).clamp_(0, 1).cpu().numpy()


# --- XDoG --------------------------------------------------------------------


def xdog(
    gray: np.ndarray,
    sigma: float = 1.0,
    k: float = 1.6,
    p: float = 30.0,
    epsilon: float = 0.28,
    phi: float = 20.0,
    tone_floor: float = 0.35,
) -> np.ndarray:
    """Extended Difference-of-Gaussians (Winnemöller et al. 2012) -> ink in [0, 1].

    `gray` is float32 in [0, 1]. Larger `epsilon` keeps more (fainter) edges.
    Gray is first lifted to [tone_floor, 1] so flat dark areas (black clothing,
    shadows) stay above `epsilon` and only real edges become ink; shadows are
    drawn by the pencil style's hatching instead of solid black blobs.
    """
    gray = tone_floor + (1.0 - tone_floor) * gray
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    d = (1 + p) * g1 - p * g2
    # Soft threshold: white above epsilon, smooth ramp into black below it.
    e = np.where(d >= epsilon, 1.0, 1.0 + np.tanh(phi * (d - epsilon)))
    return np.clip(1.0 - e, 0, 1).astype(np.float32)
