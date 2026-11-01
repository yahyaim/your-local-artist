"""Compute device selection: CUDA, then Apple MPS, then CPU."""

import contextlib
import os
from dataclasses import dataclass
from functools import cache

# Some SAM 2 ops have no MPS kernel yet; let them fall back to CPU instead of
# raising. Must be set before torch initialises the MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


@dataclass(frozen=True)
class DeviceInfo:
    kind: str  # "cuda" | "mps" | "cpu"
    name: str
    # CPU runs are slow; the UI warns and defaults to smaller output.
    is_slow: bool


@cache
def device_info() -> DeviceInfo:
    forced = os.environ.get("DEVICE")  # e.g. DEVICE=cpu to test the slow path
    if (forced in (None, "cuda")) and torch.cuda.is_available():
        return DeviceInfo("cuda", torch.cuda.get_device_name(0), False)
    if (forced in (None, "mps")) and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple Silicon GPU (MPS)", False)
    return DeviceInfo("cpu", "CPU", True)


def torch_device() -> torch.device:
    return torch.device(device_info().kind)


def free_memory() -> None:
    """Return cached GPU memory to the system between pipeline stages."""
    import gc

    gc.collect()
    kind = device_info().kind
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def sam_autocast() -> contextlib.AbstractContextManager:
    """Reduced precision for SAM 2: ~1.7x faster on Apple MPS with identical masks in
    our tests; bfloat16 on CUDA as Meta recommends. CPU stays float32."""
    kind = device_info().kind
    if kind == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if kind == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    return contextlib.nullcontext()
