# my-local-artist

A local web app that turns a video of you into an animated, stroke-drawn version, then exports it to Blender Grease Pencil. Everything runs on your machine. The only network access is the one-time model weight download.

> **Status: Milestone 1 of 5.** Upload, preview, confirm/discard and project storage work. Style previews (M2), generation (M3–M4) and SVG/Blender export (M5) are next.

## Quick start

```bash
make setup     # one time: Python 3.11 venv + backend deps, frontend deps
make dev       # starts backend (127.0.0.1:8000) and frontend (http://localhost:5173)
```

Open http://localhost:5173.

Other commands:

```bash
make test      # backend tests
make sample    # writes storage/sample_clip.mp4, a 4 s synthetic clip for testing without your own footage
```

### Requirements

- macOS or Linux
- Python 3.11. `make setup` installs it with Homebrew if it's missing, or you can point it at one with `PYTHON=/path/to/python3.11 make setup`.
- Node.js 18+
- FFmpeg (`brew install ffmpeg`)
- Blender 4.x, needed only for the `.blend` export in M5

GPU/CPU requirements and speed figures will be added in M3, once the models run.

## How it works

### 1. Upload (M1)

When you pick a file, the browser starts playing it immediately from your disk. At the same time it streams the file to `storage/temp/<upload_id>/`, and the server reads its metadata with `ffprobe`. You see resolution, fps, duration, size and codec. The browser alone can't report fps, which is why the server probe is needed.

- **Use this video** moves the file into `storage/projects/<project_id>/` and writes `project.json`.
- **Discard** deletes the temp folder.
- Abandoned temp uploads are deleted automatically once nothing in them has changed for 60 minutes (`TEMP_TTL_MINUTES`).

### Project layout

```
backend/
  app/          FastAPI app: config, storage, ffprobe, HTTP routes
  pipeline/     frame extraction, masks, flow, stabilisation      (M3+)
  styles/       the three drawing styles                          (M2+)
  export/       SVG / Blender export                              (M5)
  models/       downloaded model weights (git-ignored)
  tests/
frontend/       React + Vite + TypeScript + Tailwind
blender_export/ headless Blender script                           (M5)
storage/        temp/ uploads and projects/ (git-ignored)
scripts/        setup.sh, dev.sh, make_sample_clip.py
```

## Configuration

Set these as environment variables, or in a `.env` file at the repo root:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STORAGE_DIR` | `./storage` | Where temp uploads and projects live |
| `TEMP_TTL_MINUTES` | `60` | Unconfirmed uploads idle this long are deleted |

## Decisions and assumptions

- **Raw-body upload.** The file is sent as the raw request body, not multipart, so it streams straight to disk without being copied twice and the UI can show upload progress.
- **Preview before saving.** The preview plays from a local object URL, so it starts before the upload finishes. If your browser can't decode the codec (for example HEVC `.mov` in some browsers), the app says so and processing still works.
- **Phone rotation.** For rotated phone videos, the reported resolution is the display orientation.
- **Colours.** The UI uses the [palette](https://colorhunt.co/palette/f5f5f5dff1f1bbd5daff0000) `#F5F5F5` (page), `#DFF1F1` (panels), `#BBD5DA` (borders), `#FF0000` (primary action and selection). The palette has no dark tone, so I added `#17262B` for body text.
- **No web fonts or CDNs.** The UI uses system fonts, so the app makes no external requests.
- **Blender target.** Blender 4.5 LTS, which uses the new Grease Pencil API (GPv3, Blender 4.3+). M5 will also handle the GPv2 API for Blender 4.0–4.2.
