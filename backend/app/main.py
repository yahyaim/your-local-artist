"""FastAPI entry point: `uvicorn app.main:app` from the backend/ directory."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pipeline import weights
from pipeline.flow import raft_variant
from pipeline.subject import sam2_variant

from .config import Settings
from .jobs import JobManager
from .previews import PreviewService
from .routes import exports, jobs, preview, projects, uploads
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
log = logging.getLogger("my_local_artist")


def _run_sweeper(storage: Storage, jobs: JobManager, stop: threading.Event) -> None:
    # Check a few times per TTL, but at least every 5 minutes.
    interval = min(max(min(storage.temp_ttl_seconds, storage.project_ttl_seconds) / 4, 1.0), 300.0)
    while True:
        for removed_id in storage.sweep_expired(busy=jobs.busy_projects()):
            log.info("Removed expired upload/project %s", removed_id)
        if stop.wait(interval):
            return


def _prefetch_models() -> None:
    """First run: fetch the weights this device needs, so the first preview doesn't wait as long."""
    for key in ("person_detector", sam2_variant(), "line_contour", raft_variant(low_memory=False)):
        try:
            weights.ensure(key)
        except RuntimeError:
            pass  # already logged; requests that need it will retry and report the error


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    storage = Storage(settings.temp_dir, settings.projects_dir, settings.temp_ttl_minutes * 60,
                      settings.project_ttl_hours * 3600)
    job_manager = JobManager(storage)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()
        sweeper = threading.Thread(target=_run_sweeper, args=(storage, job_manager, stop), daemon=True)
        sweeper.start()
        if settings.prefetch_models:
            threading.Thread(target=_prefetch_models, daemon=True).start()
        yield
        stop.set()
        job_manager.shutdown()

    app = FastAPI(title="my-local-artist", lifespan=lifespan)
    app.state.settings = settings
    app.state.storage = storage
    app.state.previews = PreviewService(storage)
    app.state.jobs = job_manager
    app.include_router(uploads.router)
    app.include_router(projects.router)
    app.include_router(preview.router)
    app.include_router(jobs.router)
    app.include_router(exports.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
