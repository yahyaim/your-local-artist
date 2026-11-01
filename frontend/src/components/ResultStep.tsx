import { useEffect, useRef, useState } from "react";
import {
  buildExport, exportDownloadUrl, getRender, renderFileUrl,
  type BlenderStatus, type ExportKind, type Project, type RenderInfo,
} from "../api";

interface Props {
  project: Project;
  renderId: string;
  blender: BlenderStatus;
  onTryAnotherStyle: () => void;
  onStartOver: () => void;
}

/** Builds an export on the server (with a spinner and readable errors), then downloads it. */
function ExportButton({ project, renderId, kind, label, disabled, title }: {
  project: Project; renderId: string; kind: ExportKind; label: string; disabled?: boolean; title?: string;
}) {
  const [state, setState] = useState<{ busy: boolean; error: string | null }>({ busy: false, error: null });
  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        disabled={disabled || state.busy}
        title={title}
        onClick={async () => {
          setState({ busy: true, error: null });
          try {
            await buildExport(project.id, renderId, kind);
            window.location.href = exportDownloadUrl(project.id, renderId, kind);
            setState({ busy: false, error: null });
          } catch (e) {
            setState({ busy: false, error: (e as Error).message });
          }
        }}
        className="rounded-xl border border-steel px-4 py-2.5 font-semibold hover:bg-mist disabled:cursor-not-allowed disabled:text-ink-soft disabled:hover:bg-transparent"
      >
        {state.busy ? "Preparing…" : label}
      </button>
      {state.error && <p className="max-w-xs whitespace-pre-wrap text-xs text-signal">{state.error}</p>}
    </div>
  );
}

const STYLE_LABELS: Record<string, string> = {
  clean: "Clean cartoon outlines",
  pencil: "Sketchy pencil",
  rotoscope: "Rotoscope",
};

/**
 * Step 6: original and stroke video in sync. The original is the master
 * player (it has the controls); the stroke video follows it. The overlay
 * toggle stacks the stroke video on top with multiply blending, so its white
 * background disappears and only the lines show over the footage.
 */
export function ResultStep({ project, renderId, blender, onTryAnotherStyle, onStartOver }: Props) {
  const [info, setInfo] = useState<RenderInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overlay, setOverlay] = useState(false);
  const master = useRef<HTMLVideoElement>(null);
  const follower = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    getRender(project.id, renderId).then(setInfo).catch((e: Error) => setError(e.message));
  }, [project.id, renderId]);

  useEffect(() => {
    const a = master.current;
    const b = follower.current;
    if (!a || !b) return;
    const syncTime = () => {
      if (Math.abs(b.currentTime - a.currentTime) > 0.04) b.currentTime = a.currentTime;
    };
    const onPlay = () => {
      syncTime();
      void b.play();
    };
    const onPause = () => {
      b.pause();
      syncTime();
    };
    const onRate = () => (b.playbackRate = a.playbackRate);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("seeked", syncTime);
    a.addEventListener("ratechange", onRate);
    // Two decoders drift apart slowly; nudge the follower a few times a second.
    const drift = window.setInterval(() => !a.paused && syncTime(), 250);
    return () => {
      a.removeEventListener("play", onPlay);
      a.removeEventListener("pause", onPause);
      a.removeEventListener("seeked", syncTime);
      a.removeEventListener("ratechange", onRate);
      window.clearInterval(drift);
    };
  }, [info]);

  if (error) return <p className="font-medium text-signal">{error}</p>;
  if (!info) return <p className="text-ink-soft">Loading result…</p>;

  const aspect = `${info.width} / ${info.height}`;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold">{STYLE_LABELS[info.style] ?? info.style}</h2>
          <p className="text-sm text-ink-soft">
            {info.frames} frames · {info.width}×{info.height} · {info.fps} fps · made in {Math.round(info.seconds_taken)} s
          </p>
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm font-medium">
          <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)}
                 className="h-4 w-4 accent-signal" />
          Overlay strokes on original
        </label>
      </div>

      {/* The same two <video> elements in both layouts, so toggling never interrupts playback. */}
      <div className={overlay ? "relative mx-auto w-full max-w-3xl" : "grid gap-4 md:grid-cols-2"}>
        <figure className="flex flex-col gap-2">
          <video ref={master} src={renderFileUrl(project.id, renderId, "original.mp4")} controls playsInline preload="auto"
                 className="w-full rounded-2xl bg-ink" style={{ aspectRatio: aspect }} />
          {!overlay && <figcaption className="text-center text-sm text-ink-soft">Original</figcaption>}
        </figure>
        <figure className={overlay ? "pointer-events-none absolute inset-x-0 top-0 mix-blend-multiply" : "flex flex-col gap-2"}>
          <video ref={follower} src={renderFileUrl(project.id, renderId, "preview.mp4")} muted playsInline preload="auto"
                 className={"w-full bg-white " + (overlay ? "rounded-2xl" : "rounded-2xl ring-1 ring-steel")}
                 style={{ aspectRatio: aspect }} />
          {!overlay && <figcaption className="text-center text-sm text-ink-soft">Stroke video · use the left player&apos;s controls</figcaption>}
        </figure>
      </div>

      <div className="flex flex-wrap gap-3">
        <a href={renderFileUrl(project.id, renderId, "preview.mp4", true)}
           className="rounded-xl bg-signal px-4 py-2.5 font-semibold text-white hover:brightness-90">
          Download preview MP4
        </a>
        <ExportButton project={project} renderId={renderId} kind="svg" label="Download SVG frames (zip)" />
        <ExportButton
          project={project} renderId={renderId} kind="blend" label="Export .blend (Grease Pencil)"
          disabled={!blender.available}
          title={blender.available ? `Blender ${blender.version} (${blender.api})` : blender.error ?? undefined}
        />
        <span className="flex-1" />
        <button type="button" onClick={onTryAnotherStyle}
                className="rounded-xl border border-steel px-4 py-2.5 font-semibold hover:bg-mist">
          Try another style
        </button>
        <button type="button"
                onClick={() => {
                  if (window.confirm("Start over deletes this video and its renders from disk. Download anything you want to keep first.")) onStartOver();
                }}
                className="rounded-xl border border-steel px-4 py-2.5 font-semibold hover:bg-mist">
          Start over
        </button>
      </div>
      {!blender.available && (
        <p className="rounded-xl bg-mist p-3 text-sm">
          <strong>.blend export is off:</strong> {blender.error} Everything else works without Blender.
        </p>
      )}
    </div>
  );
}
