"""Background generation jobs: a single worker thread, progress snapshots, cancel, ETA.

One worker on purpose: every job wants the whole GPU, so running two at once
would only make both slower and risk running out of memory. Extra jobs queue.

After every job (finished, cancelled or failed) the project's per-frame caches
are deleted, so only the render itself stays on disk.
"""

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.device import device_info
from pipeline.frames import output_size
from pipeline.generate import STAGES, Cancelled, GenerateSettings, SubjectSpec, clear_intermediates, generate

from .storage import Storage

log = logging.getLogger("my_local_artist.jobs")

# Rough seconds per unit of work for a 1280x720 frame on a GPU. Only used for
# the ETA until a stage has measured its own speed; then the measured/guessed
# ratio of finished stages recalibrates the guesses for the stages still to come.
_GUESS_S_PER_UNIT = {"frames": 0.01, "masking": 0.5, "lines": 0.2, "flow": 0.25, "vectorize": 0.08,
                     "smoothing": 0.05, "encoding": 0.03}
_CPU_SLOWDOWN = 6.0


@dataclass
class Stage:
    key: str
    label: str
    status: str = "pending"  # pending | running | done | cached | skipped
    done: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class Job:
    id: str
    project_id: str
    style: str
    settings: GenerateSettings
    subject: SubjectSpec
    est_frames: int
    pixel_scale: float  # output pixels relative to 1280x720
    render_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"  # queued | running | done | failed | cancelled
    stages: list[Stage] = field(default_factory=lambda: [Stage(k, label) for k, label in STAGES])
    error: str | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def _guess(self, stage: Stage) -> float:
        units = stage.total or self.est_frames * (2 if stage.key == "encoding" else 1)
        slow = _CPU_SLOWDOWN if device_info().is_slow else 1.0
        return _GUESS_S_PER_UNIT[stage.key] * units * self.pixel_scale * slow

    def _calibration(self) -> float:
        done = [s for s in self.stages if s.status == "done" and s.total]
        guessed = sum(self._guess(s) for s in done)
        actual = sum(s.finished_at - s.started_at for s in done)
        return actual / guessed if guessed > 0.5 else 1.0

    def eta_and_progress(self) -> tuple[float | None, float]:
        if self.status in ("done", "failed", "cancelled"):
            return 0.0, 1.0 if self.status == "done" else 0.0
        now, calib = time.time(), self._calibration()
        remaining = total_w = done_w = 0.0
        for s in self.stages:
            guess = self._guess(s)
            if s.status in ("cached", "skipped"):
                continue
            total_w += guess
            frac = s.done / s.total if s.total else 0.0
            if s.status == "done":
                done_w += guess
            elif s.status == "running":
                done_w += guess * frac
                if s.done >= 2:
                    remaining += (now - s.started_at) / s.done * (s.total - s.done)
                else:
                    remaining += guess * calib * (1 - frac)
            else:
                remaining += guess * calib
        progress = done_w / total_w if total_w else 0.0
        return (remaining if self.status == "running" else None), progress

    def snapshot(self) -> dict:
        eta, progress = self.eta_and_progress()
        current = next((s for s in self.stages if s.status == "running"), None)
        return {
            "id": self.id,
            "project_id": self.project_id,
            "style": self.style,
            "status": self.status,
            "render_id": self.render_id if self.status == "done" else None,
            "stage": current.key if current else None,
            "stages": [
                {"key": s.key, "label": s.label, "status": s.status,
                 "progress": (s.done / s.total) if s.total else (1.0 if s.status != "pending" else 0.0)}
                for s in self.stages
            ],
            "progress": round(progress, 4),
            "eta_s": round(eta, 1) if eta is not None else None,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else 0,
            "error": self.error,
        }


class _Reporter:
    def __init__(self, job: Job) -> None:
        self.job = job
        self.current: Stage | None = None

    def _stage(self, key: str) -> Stage:
        return next(s for s in self.job.stages if s.key == key)

    def begin(self, stage: str, total: int) -> None:
        s = self._stage(stage)
        s.status, s.total, s.done, s.started_at = "running", max(total, 1), 0, time.time()
        self.current = s

    def advance(self, done: int) -> None:
        if self.current is not None:
            # ffmpeg's frame estimate can be off by a frame or two; never exceed 100%.
            self.current.done = min(done, self.current.total)

    def finish(self, stage: str, status: str = "done") -> None:
        s = self._stage(stage)
        s.status, s.finished_at = status, time.time()
        if status == "done":
            s.done = s.total
        self.current = None

    def check_cancel(self) -> None:
        if self.job.cancel_event.is_set():
            raise Cancelled()


class JobManager:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generate")

    def submit(self, project_id: str, style: str, settings: GenerateSettings, subject: SubjectSpec) -> Job:
        project = self.storage.get_project(project_id)
        meta = project.metadata
        fps = meta.fps / (2 if settings.half_fps else 1)
        end = settings.end_s if settings.end_s is not None else meta.duration_s
        w, h = output_size(meta.width, meta.height, settings.max_side)
        job = Job(
            id=uuid.uuid4().hex, project_id=project_id, style=style, settings=settings, subject=subject,
            est_frames=max(1, int((min(end, meta.duration_s) - settings.start_s) * fps)),
            pixel_scale=(w * h) / (1280 * 720),
        )
        self.jobs[job.id] = job
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        job.cancel_event.set()
        if job.status == "queued":
            job.status, job.finished_at = "cancelled", time.time()

    def busy_projects(self) -> frozenset[str]:
        return frozenset(j.project_id for j in self.jobs.values() if j.status in ("queued", "running"))

    def cancel_project(self, project_id: str, timeout: float = 30.0) -> bool:
        """Cancel the project's jobs and wait until none is running. False on timeout."""
        jobs = [j for j in self.jobs.values() if j.project_id == project_id]
        for job in jobs:
            self.cancel(job.id)
        deadline = time.time() + timeout
        while any(j.status == "running" for j in jobs):
            if time.time() > deadline:
                return False
            time.sleep(0.05)
        return True

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job) -> None:
        if job.status == "cancelled":
            return
        job.status, job.started_at = "running", time.time()
        project = self.storage.get_project(job.project_id)
        project_dir = self.storage.project_dir(job.project_id)
        # The final status is published only after cleanup, so whoever sees a job
        # finish (the UI, "Start over") also sees its files gone.
        status, error = "failed", None
        try:
            job.result = generate(
                project_dir=project_dir,
                source=project_dir / project.source_file,
                metadata=project.metadata.model_dump(),
                subject=job.subject,
                style_key=job.style,
                settings=job.settings,
                render_id=job.render_id,
                rep=_Reporter(job),
            )
            status = "done"
            log.info("Job %s finished in %.1fs", job.id, time.time() - job.started_at)
        except Cancelled:
            status = "cancelled"
            log.info("Job %s cancelled", job.id)
        except Exception as exc:  # report any failure to the UI rather than dying silently
            error = str(exc) or exc.__class__.__name__
            log.exception("Job %s failed", job.id)
        finally:
            for s in job.stages:
                if s.status == "running":
                    s.status = "pending"
            if status != "done":
                shutil.rmtree(project_dir / "renders" / job.render_id, ignore_errors=True)
            clear_intermediates(project_dir)
            job.finished_at = time.time()
            job.status, job.error = status, error
