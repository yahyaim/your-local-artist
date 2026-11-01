"""Subject selection on a single frame: auto-detect the largest person, then SAM 2."""

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from . import weights
from .device import device_info, sam_autocast, torch_device

# Serialises GPU work across request threads; models aren't re-entrant and
# running them concurrently on one consumer GPU only adds memory pressure.
gpu_lock = threading.RLock()

SAM2_CONFIGS = {
    "sam2_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
}


def sam2_variant(low_memory: bool = False) -> str:
    """Small by default; tiny on CPU or in low-memory mode."""
    return "sam2_tiny" if low_memory or device_info().kind == "cpu" else "sam2_small"


@dataclass
class Prompt:
    """What the user (or auto-detection) selected, in frame pixel coordinates."""

    box: list[float] | None = None  # [x0, y0, x1, y1]
    points: list[list[float]] = field(default_factory=list)  # [[x, y, 1=include|0=exclude], ...]

    @property
    def is_empty(self) -> bool:
        return self.box is None and not self.points


_detector = None
_image_predictors: dict[str, object] = {}
_load_lock = threading.Lock()


def _person_detector():
    global _detector
    with _load_lock:
        if _detector is None:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

            model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=91)
            model.load_state_dict(torch.load(weights.ensure("person_detector"), map_location="cpu", weights_only=True))
            # Detection runs once per preview frame: CPU is fast enough and avoids
            # MPS gaps in torchvision's detection ops.
            _detector = model.eval()
        return _detector


def _image_predictor(variant: str):
    with _load_lock:
        if variant not in _image_predictors:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(SAM2_CONFIGS[variant], str(weights.ensure(variant)), device=str(torch_device()))
            _image_predictors[variant] = SAM2ImagePredictor(model)
        return _image_predictors[variant]


def release_models() -> None:
    global _detector
    with _load_lock:
        _detector = None
        _image_predictors.clear()


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


@torch.inference_mode()
def detect_people(rgb: np.ndarray, min_score: float = 0.6) -> list[list[float]]:
    """Boxes [x0, y0, x1, y1] of confidently detected people, largest first."""
    with gpu_lock:
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        det = _person_detector()([x])[0]
    boxes = det["boxes"][(det["labels"] == 1) & (det["scores"] >= min_score)]  # COCO label 1 = person
    return sorted(boxes.tolist(), key=_area, reverse=True)


def auto_prompt(rgb: np.ndarray) -> Prompt:
    """Largest person; falls back to a point at the frame centre if nobody is found."""
    people = detect_people(rgb)
    if people:
        return Prompt(box=people[0])
    h, w = rgb.shape[:2]
    return Prompt(points=[[w / 2, h / 2, 1]])


def click_prompt(rgb: np.ndarray, points: list[list[float]]) -> Prompt:
    """Prompt from user clicks.

    A lone click on a shirt is ambiguous to SAM (shirt or whole person?). If the
    first include-click lands inside a detected person, that person's box is
    added so the click selects the whole person; clicks elsewhere (a pet, a
    prop) are used as plain points.
    """
    include = next((p for p in points if p[2] == 1), None)
    box = None
    if include is not None:
        hits = [b for b in detect_people(rgb) if b[0] <= include[0] <= b[2] and b[1] <= include[1] <= b[3]]
        box = min(hits, key=_area) if hits else None
    return Prompt(box=box, points=points)


def clean_mask(mask: np.ndarray, min_fraction: float = 0.02) -> np.ndarray:
    """Drop specks and fill pinholes, so outlines don't trace segmentation noise."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(areas.max() * min_fraction, 16)
    m = keep[labels].astype(np.uint8)
    # Fill holes smaller than ~0.1% of the frame (real gaps, e.g. between arm and body, survive).
    inv = 1 - m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < mask.size * 0.001
    m[small[labels]] = 1
    return m.astype(bool)


@torch.inference_mode()
def segment(rgb: np.ndarray, prompt: Prompt, low_memory: bool = False) -> np.ndarray:
    """Boolean subject mask for one frame."""
    with gpu_lock, sam_autocast():
        predictor = _image_predictor(sam2_variant(low_memory))
        predictor.set_image(rgb)
        box = np.asarray(prompt.box, np.float32) if prompt.box is not None else None
        coords = labels = None
        if prompt.points:
            pts = np.asarray(prompt.points, np.float32)
            coords, labels = pts[:, :2], pts[:, 2].astype(np.int32)
        # A single click is ambiguous (shirt vs. whole person): ask for 3 candidates
        # and keep the best-scoring one, as SAM's authors recommend.
        multimask = box is None and coords is not None and len(coords) == 1
        masks, scores, _ = predictor.predict(
            point_coords=coords, point_labels=labels, box=box, multimask_output=multimask
        )
        predictor.reset_predictor()
    return clean_mask(masks[int(np.argmax(scores))] > 0)
