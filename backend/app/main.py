"""FastAPI entry point: `uvicorn app.main:app` from the backend/ directory."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import Settings
from .routes import projects, uploads
from .storage import Storage

log = logging.getLogger("my_local_artist")


def _run_sweeper(storage: Storage, stop: threading.Event) -> None:
    # Check a few times per TTL, but at least every 5 minutes.
    interval = min(max(storage.temp_ttl_seconds / 4, 1.0), 300.0)
    while True:
        for upload_id in storage.sweep_expired():
            log.info("Removed expired temp upload %s", upload_id)
        if stop.wait(interval):
            return


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    storage = Storage(settings.temp_dir, settings.projects_dir, settings.temp_ttl_minutes * 60)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()
        sweeper = threading.Thread(target=_run_sweeper, args=(storage, stop), daemon=True)
        sweeper.start()
        yield
        stop.set()

    app = FastAPI(title="my-local-artist", lifespan=lifespan)
    app.state.settings = settings
    app.state.storage = storage
    app.include_router(uploads.router)
    app.include_router(projects.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
