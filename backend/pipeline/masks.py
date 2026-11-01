"""Subject masks for every frame: SAM 2 video propagation, in bounded-memory chunks.

SAM 2's video predictor keeps every loaded frame (resized to 1024², float32,
~12 MB each) plus per-frame memory features. To stay within a consumer GPU,
frames are processed in chunks: each chunk is a folder of symlinks, seeded
with the previous chunk's last mask. Propagating backwards from the prompt
frame reuses the same forward code on a reversed chunk.
"""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from . import weights
from .device import free_memory, sam_autocast, torch_device
from .subject import SAM2_CONFIGS, Prompt, clean_mask, gpu_lock


def _build_video_predictor(variant: str):
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(SAM2_CONFIGS[variant], str(weights.ensure(variant)), device=str(torch_device()))


@torch.inference_mode()
def _propagate_chunk(
    predictor, frames: list[Path], seed: Prompt | np.ndarray, low_memory: bool,
    on_mask: Callable[[int, np.ndarray], None],
) -> np.ndarray:
    """Track the subject from frames[0] forward; returns the last frame's mask."""
    with tempfile.TemporaryDirectory(prefix="sam2_chunk_") as tmp:
        for i, frame in enumerate(frames):
            os.symlink(frame.resolve(), Path(tmp) / f"{i}.jpg")
        state = predictor.init_state(
            video_path=tmp, offload_video_to_cpu=low_memory, offload_state_to_cpu=low_memory
        )
        if isinstance(seed, np.ndarray):
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(seed))
        else:
            pts = np.asarray(seed.points, np.float32) if seed.points else None
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=pts[:, :2] if pts is not None else None,
                labels=pts[:, 2].astype(np.int32) if pts is not None else None,
                box=np.asarray(seed.box, np.float32) if seed.box is not None else None,
            )
        last = None
        for idx, _, logits in predictor.propagate_in_video(state):
            last = clean_mask((logits[0, 0] > 0).cpu().numpy())
            on_mask(idx, last)
        predictor.reset_state(state)
        del state
    assert last is not None
    return last


def propagate_masks(
    frames: list[Path],
    prompt_index: int,
    prompt: Prompt,
    out_dir: Path,
    variant: str,
    low_memory: bool,
    on_progress: Callable[[int], None],
    check_cancel: Callable[[], None],
) -> None:
    """Write `out_dir/<index>.png` (0/255) for every frame, tracking from `prompt_index` both ways."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = 60 if low_memory else 150
    done: set[int] = set()

    def run(order: list[int]) -> None:
        # `order` starts at the prompt frame; chunks overlap by one frame,
        # which is re-seeded with the mask tracked so far.
        seed: Prompt | np.ndarray = prompt
        start = 0
        while start < len(order) - 1 or (start == 0 and len(order) == 1):
            check_cancel()
            ids = order[start:start + chunk]

            def save(local_idx: int, mask: np.ndarray, ids: list[int] = ids) -> None:
                frame_idx = ids[local_idx]
                cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), mask.astype(np.uint8) * 255)
                if frame_idx not in done:
                    done.add(frame_idx)
                    on_progress(len(done))
                check_cancel()  # raising here abandons the chunk; its temp folder is cleaned up

            with gpu_lock, sam_autocast():
                seed = _propagate_chunk(predictor, [frames[i] for i in ids], seed, low_memory, save)
            free_memory()
            if len(ids) == 1:
                break
            start += len(ids) - 1

    predictor = _build_video_predictor(variant)
    try:
        run(list(range(prompt_index, len(frames))))
        if prompt_index > 0:
            run(list(range(prompt_index, -1, -1)))
    finally:
        del predictor
        free_memory()
