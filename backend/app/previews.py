"""Single-frame previews: frame grab, subject selection and style samples, cached per project.

Cache layout (inside the project folder):
    subject.json                       current subject prompt (normalised coordinates)
    cache/preview/frame_<side>_<ms>.png
    cache/preview/mask_<key>.png       keyed by frame + prompt
    cache/preview/ink_<key>.npy        neural line map, shared by clean + rotoscope
    cache/preview/style_<key>.png      rendered style sample
"""

import hashlib
import json
import threading
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from pipeline import subject as subj
from pipeline.frames import output_size, read_frame
from pipeline.stylize import neural_ink, stylize_frame
from styles import STYLES, StyleParams

from .storage import Project, Storage


class SubjectState(BaseModel):
    time: float
    mode: str  # "auto" | "click"
    # Normalised [0, 1] frame coordinates so the prompt survives resolution changes.
    box: list[float] | None = None
    points: list[list[float]] = Field(default_factory=list)  # [[x, y, 1|0], ...]

    @property
    def version(self) -> str:
        return hashlib.sha1(self.model_dump_json().encode()).hexdigest()[:12]

    def to_prompt(self, w: int, h: int) -> subj.Prompt:
        box = [self.box[0] * w, self.box[1] * h, self.box[2] * w, self.box[3] * h] if self.box else None
        return subj.Prompt(box=box, points=[[x * w, y * h, lab] for x, y, lab in self.points])

    @classmethod
    def from_prompt(cls, prompt: subj.Prompt, time: float, mode: str, w: int, h: int) -> "SubjectState":
        box = [prompt.box[0] / w, prompt.box[1] / h, prompt.box[2] / w, prompt.box[3] / h] if prompt.box else None
        return cls(time=time, mode=mode, box=box, points=[[x / w, y / h, lab] for x, y, lab in prompt.points])


def _key(*parts: object) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class PreviewService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        # One lock per project: the browser asks for all three style samples at
        # once, and they share the frame, mask and line map - compute those once.
        self._locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)

    # --- helpers --------------------------------------------------------------

    def _cache_dir(self, project_id: str) -> Path:
        d = self.storage.project_dir(project_id) / "cache" / "preview"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def clamp_time(project: Project, t: float) -> float:
        last = max(project.metadata.duration_s - 1.0 / project.metadata.fps, 0.0)
        return round(min(max(t, 0.0), last), 3)

    @staticmethod
    def size(project: Project, max_side: int) -> tuple[int, int]:
        return output_size(project.metadata.width, project.metadata.height, max_side)

    # --- frame ----------------------------------------------------------------

    def frame(self, project_id: str, t: float, max_side: int) -> np.ndarray:
        project = self.storage.get_project(project_id)
        t = self.clamp_time(project, t)
        path = self._cache_dir(project_id) / f"frame_{max_side}_{int(t * 1000)}.png"
        with self._locks[project_id]:
            if path.is_file():
                return cv2.imread(str(path))[..., ::-1].copy()
            rgb = read_frame(self.storage.project_source(project_id), t, self.size(project, max_side))
            cv2.imwrite(str(path), rgb[..., ::-1])
            return rgb

    # --- subject --------------------------------------------------------------

    def _subject_path(self, project_id: str) -> Path:
        return self.storage.project_dir(project_id) / "subject.json"

    def saved_subject(self, project_id: str) -> SubjectState | None:
        path = self._subject_path(project_id)
        return SubjectState.model_validate_json(path.read_text()) if path.is_file() else None

    def subject(self, project_id: str, t: float, max_side: int) -> SubjectState:
        """Subject for frame `t`. Moving to another frame re-runs auto-detection there."""
        project = self.storage.get_project(project_id)
        t = self.clamp_time(project, t)
        with self._locks[project_id]:
            path = self._subject_path(project_id)
            if path.is_file():
                state = SubjectState.model_validate_json(path.read_text())
                if abs(state.time - t) < 1e-3:
                    return state
            return self.set_subject(project_id, t, [], max_side)

    def set_subject(self, project_id: str, t: float, points: list[list[float]], max_side: int) -> SubjectState:
        """Select the subject from clicks (normalised [x, y, 1|0]); no clicks means auto-detect."""
        project = self.storage.get_project(project_id)
        t = self.clamp_time(project, t)
        with self._locks[project_id]:
            rgb = self.frame(project_id, t, max_side)
            h, w = rgb.shape[:2]
            if points:
                prompt = subj.click_prompt(rgb, [[x * w, y * h, int(lab)] for x, y, lab in points])
                state = SubjectState.from_prompt(prompt, t, "click", w, h)
            else:
                state = SubjectState.from_prompt(subj.auto_prompt(rgb), t, "auto", w, h)
            self._subject_path(project_id).write_text(state.model_dump_json(indent=2))
            return state

    def mask(self, project_id: str, t: float, max_side: int) -> np.ndarray:
        with self._locks[project_id]:
            state = self.subject(project_id, t, max_side)
            path = self._cache_dir(project_id) / f"mask_{_key(state.version, max_side)}.png"
            if path.is_file():
                return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            rgb = self.frame(project_id, state.time, max_side)
            h, w = rgb.shape[:2]
            mask = subj.segment(rgb, state.to_prompt(w, h))
            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            return mask

    def mask_overlay_png(self, project_id: str, t: float, max_side: int) -> bytes:
        """RGBA overlay for the UI: background dimmed, subject clear, red outline."""
        mask = self.mask(project_id, t, max_side)
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), np.uint8)
        overlay[~mask] = (23, 38, 43, 150)  # ink colour, translucent
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0, 255), max(2, round(max(h, w) / 360)), cv2.LINE_AA)
        ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
        return png.tobytes()

    # --- styles ---------------------------------------------------------------

    def style_preview_png(self, project_id: str, style: str, t: float, max_side: int, params: StyleParams) -> bytes:
        with self._locks[project_id]:
            state = self.subject(project_id, t, max_side)
            out = self._cache_dir(project_id) / f"style_{_key(style, state.version, max_side, params.cache_key())}.png"
            if out.is_file():
                return out.read_bytes()
            rgb = self.frame(project_id, state.time, max_side)
            mask = self.mask(project_id, state.time, max_side)
            ink = None
            if STYLES[style].needs_neural_lines:
                ink_path = self._cache_dir(project_id) / f"ink_{_key(state.version, max_side, params.remove_background)}.npy"
                if ink_path.is_file():
                    ink = np.load(ink_path).astype(np.float32)
                else:
                    ink = neural_ink(rgb, mask, params)
                    np.save(ink_path, ink.astype(np.float16))
        # Vectorising and rendering is CPU-only; let other styles proceed meanwhile.
        _, image = stylize_frame(style, rgb, mask, params, ink)
        ok, png = cv2.imencode(".png", image[..., ::-1])
        out.write_bytes(png.tobytes())
        return png.tobytes()
