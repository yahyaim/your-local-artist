"""Model weight registry and on-demand download into backend/models/.

Downloads are the app's only network access. Each file is fetched once,
verified against a pinned SHA-256, and its progress is exposed so the UI can
show what's happening on first run.
"""

import hashlib
import logging
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("my_local_artist.weights")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class WeightFile:
    key: str
    label: str
    url: str
    filename: str
    sha256: str
    size_mb: float


_INFORMATIVE_DRAWINGS_REV = "bd4b4299be505803e036203a39c02024b4cfee11"

WEIGHTS: dict[str, WeightFile] = {
    w.key: w
    for w in [
        WeightFile(
            key="sam2_small",
            label="SAM 2.1 (small)",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            filename="sam2.1_hiera_small.pt",
            sha256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
            size_mb=184,
        ),
        WeightFile(
            key="sam2_tiny",
            label="SAM 2.1 (tiny)",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            filename="sam2.1_hiera_tiny.pt",
            sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
            size_mb=156,
        ),
        WeightFile(
            key="line_contour",
            label="Informative Drawings (contour lines)",
            url=f"https://huggingface.co/spaces/carolineec/informativedrawings/resolve/{_INFORMATIVE_DRAWINGS_REV}/model.pth",
            filename="informative_drawings_contour.pth",
            sha256="c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
            size_mb=17,
        ),
        WeightFile(
            key="raft_large",
            label="RAFT optical flow (large)",
            url="https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
            filename="raft_large_C_T_SKHT_V2-ff5fadd5.pth",
            sha256="ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322",
            size_mb=21,
        ),
        WeightFile(
            key="raft_small",
            label="RAFT optical flow (small)",
            url="https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth",
            filename="raft_small_C_T_V2-01064c6d.pth",
            sha256="01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27",
            size_mb=4,
        ),
        WeightFile(
            key="person_detector",
            label="Person detector (Faster R-CNN MobileNetV3)",
            url="https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
            filename="fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
            sha256="fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533",
            size_mb=78,
        ),
    ]
}


@dataclass
class DownloadState:
    status: str = "missing"  # missing | downloading | ready | error
    done_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None
    finished: threading.Event = field(default_factory=threading.Event)


_states: dict[str, DownloadState] = {key: DownloadState() for key in WEIGHTS}
_lock = threading.Lock()


def weight_path(key: str) -> Path:
    return MODELS_DIR / WEIGHTS[key].filename


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(w: WeightFile, state: DownloadState) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / w.filename
    part = target.with_suffix(target.suffix + ".part")
    log.info("Downloading %s (~%.0f MB) from %s", w.label, w.size_mb, w.url)
    req = urllib.request.Request(w.url, headers={"User-Agent": "my-local-artist"})
    with urllib.request.urlopen(req, timeout=60) as res, part.open("wb") as f:
        state.total_bytes = int(res.headers.get("Content-Length") or w.size_mb * 1e6)
        last_logged = 0
        while chunk := res.read(1 << 20):
            f.write(chunk)
            state.done_bytes += len(chunk)
            if state.done_bytes - last_logged >= 25 << 20:
                last_logged = state.done_bytes
                log.info("  %s: %d / %d MB", w.label, state.done_bytes >> 20, state.total_bytes >> 20)
    digest = _sha256(part)
    if digest != w.sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {w.filename} (got {digest})")
    part.rename(target)
    log.info("Downloaded %s", w.label)


def ensure(key: str) -> Path:
    """Return the local path for a weight file, downloading it first if needed.

    Safe to call from several threads: one downloads, the others wait.
    """
    path = weight_path(key)
    state = _states[key]
    with _lock:
        if path.is_file():
            state.status = "ready"
            state.finished.set()
            return path
        owner = state.status != "downloading"
        if owner:
            state.status, state.done_bytes, state.error = "downloading", 0, None
            state.finished.clear()
    if not owner:
        state.finished.wait()
        if state.status != "ready":
            raise RuntimeError(f"Downloading {WEIGHTS[key].label} failed: {state.error}")
        return path
    try:
        _download(WEIGHTS[key], state)
        state.status = "ready"
        return path
    except Exception as exc:
        state.status, state.error = "error", str(exc)
        log.error("Downloading %s failed: %s", WEIGHTS[key].label, exc)
        raise RuntimeError(f"Downloading {WEIGHTS[key].label} failed: {exc}") from exc
    finally:
        state.finished.set()


def status() -> list[dict]:
    out = []
    for key, w in WEIGHTS.items():
        state = _states[key]
        if state.status == "missing" and weight_path(key).is_file():
            state.status = "ready"
        out.append({
            "key": key,
            "label": w.label,
            "status": state.status,
            "done_mb": round(state.done_bytes / 1e6, 1),
            "total_mb": round(state.total_bytes / 1e6, 1) if state.total_bytes else w.size_mb,
            "error": state.error,
        })
    return out
