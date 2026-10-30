import { useEffect, useRef, useState } from "react";
import { confirmUpload, discardUpload, uploadVideo, type Project, type UploadResult } from "../api";
import { MetadataList } from "./MetadataList";

const ACCEPT = ".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm";
const ALLOWED = /\.(mp4|mov|webm)$/i;

type UploadState =
  | { kind: "idle" }
  | { kind: "uploading"; progress: number }
  | { kind: "ready"; result: UploadResult }
  | { kind: "failed"; message: string };

/**
 * Step 1: pick a file, preview it, then "Use this video" or "Discard".
 * The preview plays from a local object URL immediately, while the file uploads
 * to the backend's temp folder in parallel; it only becomes a project on confirm.
 */
export function UploadStep({ onConfirmed }: { onConfirmed: (project: Project) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadState>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cannotPlay, setCannotPlay] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  function selectFile(selected: File) {
    if (!ALLOWED.test(selected.name)) {
      setError("Please choose an mp4, mov or webm file.");
      return;
    }
    setError(null);
    setCannotPlay(false);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setUpload({ kind: "uploading", progress: 0 });

    const { done, abort } = uploadVideo(selected, (progress) =>
      setUpload((prev) => (prev.kind === "uploading" ? { kind: "uploading", progress } : prev)),
    );
    abortRef.current = abort;
    done
      .then((result) => setUpload({ kind: "ready", result }))
      .catch((err: Error) => {
        if (err.name !== "AbortError") setUpload({ kind: "failed", message: err.message });
      })
      .finally(() => (abortRef.current = null));
  }

  async function discard() {
    abortRef.current?.();
    if (upload.kind === "ready") {
      // Best effort: the server's TTL sweep removes it anyway if this fails.
      await discardUpload(upload.result.upload_id).catch(() => undefined);
    }
    setFile(null);
    setPreviewUrl(null);
    setUpload({ kind: "idle" });
    setError(null);
  }

  async function confirm() {
    if (upload.kind !== "ready") return;
    setBusy(true);
    setError(null);
    try {
      onConfirmed(await confirmUpload(upload.result.upload_id));
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  if (!file || !previewUrl) {
    return (
      <section className="mx-auto max-w-3xl">
        <h2 className="mb-1 text-2xl font-semibold">Upload a video</h2>
        <p className="mb-6 text-ink-soft">
          You'll preview it first. Nothing is saved until you click “Use this video”.
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const dropped = e.dataTransfer.files[0];
            if (dropped) selectFile(dropped);
          }}
          className={
            "flex h-72 w-full flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed transition-colors " +
            (dragging ? "border-signal bg-white" : "border-steel bg-mist hover:border-ink-soft")
          }
        >
          <svg viewBox="0 0 48 48" className="h-12 w-12 text-ink-soft" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M8 34c6-2 10-14 16-14s8 8 16 6" strokeLinecap="round" />
            <path d="M24 6v18M17 13l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-lg font-medium">Drop a video here, or click to choose</span>
          <span className="text-sm text-ink-soft">mp4 · mov · webm</span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const picked = e.target.files?.[0];
            e.target.value = "";
            if (picked) selectFile(picked);
          }}
        />
        {error && <p className="mt-4 text-sm font-medium text-signal">{error}</p>}
      </section>
    );
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="overflow-hidden rounded-2xl bg-ink">
        <video
          src={previewUrl}
          controls
          playsInline
          className="max-h-[70vh] w-full"
          onError={() => setCannotPlay(true)}
        />
        {cannotPlay && (
          <p className="bg-mist p-3 text-sm">
            Your browser can't play this codec, but the file can still be processed.
          </p>
        )}
      </div>

      <aside className="flex flex-col gap-5 rounded-2xl border border-steel bg-white p-5">
        <div>
          <p className="truncate font-semibold" title={file.name}>{file.name}</p>
          <p className="text-xs text-ink-soft">Not saved yet: preview only</p>
        </div>

        {upload.kind === "uploading" && (
          <div>
            <div className="mb-1 flex justify-between text-sm">
              <span>{upload.progress < 1 ? "Uploading…" : "Reading video info…"}</span>
              <span className="tabular-nums">{Math.round(upload.progress * 100)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-mist">
              <div className="h-full bg-ink transition-[width]" style={{ width: `${upload.progress * 100}%` }} />
            </div>
          </div>
        )}
        {upload.kind === "ready" && <MetadataList metadata={upload.result.metadata} />}
        {upload.kind === "failed" && <p className="text-sm font-medium text-signal">{upload.message}</p>}
        {error && <p className="text-sm font-medium text-signal">{error}</p>}

        <div className="mt-auto flex flex-col gap-2">
          <button
            type="button"
            onClick={confirm}
            disabled={upload.kind !== "ready" || busy}
            className="rounded-xl bg-signal px-4 py-3 font-semibold text-white transition hover:brightness-90 disabled:cursor-not-allowed disabled:bg-steel disabled:text-ink-soft"
          >
            {busy ? "Saving…" : "Use this video"}
          </button>
          <button
            type="button"
            onClick={discard}
            disabled={busy}
            className="rounded-xl border border-steel px-4 py-3 font-semibold transition hover:bg-mist"
          >
            Discard
          </button>
        </div>
      </aside>
    </section>
  );
}
