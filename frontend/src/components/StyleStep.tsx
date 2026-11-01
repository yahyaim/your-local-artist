import { useEffect, useRef, useState } from "react";
import {
  frameUrl, getStyles, getSubject, setSubject, stylePreviewUrl, subjectOverlayUrl,
  type GenerateSettings, type LookSettings, type Project, type StyleInfo, type Subject, type SubjectPoint,
} from "../api";
import { formatDuration } from "../lib/format";
import { useDebounced } from "../lib/useDebounced";
import { LoadingImage } from "./LoadingImage";
import { SettingsPanel } from "./SettingsPanel";

interface Props {
  project: Project;
  settings: GenerateSettings;
  onSettingsChange: (settings: GenerateSettings) => void;
  selectedStyle: string | null;
  onSelectStyle: (style: string) => void;
  onGenerate: () => Promise<void>;
  onStartOver: () => void;
}

/**
 * Step 3: pick the subject (click-to-select, auto-detected by default) on a
 * frame chosen with the slider, and compare the three styles on that frame.
 */
export function StyleStep({
  project, settings, onSettingsChange, selectedStyle, onSelectStyle, onGenerate, onStartOver,
}: Props) {
  const { duration_s, width, height } = project.metadata;
  // Previews follow the settings, debounced so dragging a slider doesn't flood the backend.
  const maxSide = useDebounced(settings.max_side, 400);
  const look: LookSettings = useDebounced(
    {
      thickness: settings.thickness, detail: settings.detail, smoothing: settings.smoothing,
      remove_background: settings.remove_background, paper_texture: settings.paper_texture,
    },
    400,
  );
  const [starting, setStarting] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [time, setTime] = useState(Math.min(1, duration_s / 2));
  const [sliderTime, setSliderTime] = useState(time);
  const [subject, setSubjectState] = useState<Subject | null>(null);
  const [styles, setStyles] = useState<StyleInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStyles().then(setStyles).catch((e: Error) => setError(e.message));
  }, []);

  // Changing the frame re-runs auto-detection there (see README: subject selection).
  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    setError(null);
    getSubject(project.id, time, maxSide)
      .then((s) => !cancelled && setSubjectState(s))
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setBusy(false));
    return () => {
      cancelled = true;
    };
  }, [project.id, time, maxSide]);

  // Debounce the slider so dragging doesn't request a frame per pixel.
  const debounce = useRef<number | undefined>(undefined);
  function onSlide(value: number) {
    setSliderTime(value);
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => setTime(value), 350);
  }

  async function updatePoints(points: SubjectPoint[]) {
    if (!subject) return;
    setBusy(true);
    setError(null);
    try {
      setSubjectState(await setSubject(project.id, subject.time, points, maxSide));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function onFrameClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!subject || busy) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const existing = subject.mode === "click" ? subject.points : [];
    void updatePoints([...existing, [x, y, e.altKey ? 0 : 1]]);
  }

  return (
    <div className="flex flex-col gap-8">
      <section className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div
          className="relative cursor-crosshair select-none overflow-hidden rounded-2xl bg-ink"
          style={{ aspectRatio: `${width} / ${height}` }}
          onClick={onFrameClick}
        >
          <div className="absolute inset-0">
            <LoadingImage src={frameUrl(project.id, time, maxSide)} alt="Selected frame" className="h-full" />
          </div>
          {subject && (
            <img
              src={subjectOverlayUrl(project.id, subject, maxSide)}
              alt=""
              draggable={false}
              className="absolute inset-0 h-full w-full object-contain"
            />
          )}
          {subject?.mode === "click" &&
            subject.points.map(([x, y, include], i) => (
              <span
                key={i}
                className={
                  "absolute h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white " +
                  (include ? "bg-signal" : "bg-ink")
                }
                style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
              />
            ))}
          {busy && (
            <span className="absolute right-3 top-3 rounded-full bg-white/90 px-3 py-1 text-xs font-medium">
              Finding subject…
            </span>
          )}
        </div>

        <aside className="flex flex-col gap-5 rounded-2xl border border-steel bg-white p-5">
          <div>
            <h2 className="text-lg font-semibold">Subject</h2>
            <p className="mt-1 text-sm text-ink-soft">
              {subject?.mode === "click"
                ? "Using your clicks."
                : "Auto-detected: the largest person in the frame."}{" "}
              Click the subject to select it; hold Alt/⌥ and click to exclude an area.
            </p>
          </div>
          <button
            type="button"
            disabled={busy || subject?.mode !== "click"}
            onClick={() => void updatePoints([])}
            className="rounded-xl border border-steel px-4 py-2 text-sm font-semibold transition hover:bg-mist disabled:opacity-50"
          >
            Reset to auto-detect
          </button>

          <label className="flex flex-col gap-2 text-sm">
            <span className="flex justify-between">
              <span className="font-medium">Preview frame</span>
              <span className="tabular-nums text-ink-soft">{formatDuration(sliderTime)}</span>
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(duration_s - 0.05, 0)}
              step={0.05}
              value={sliderTime}
              onChange={(e) => onSlide(Number(e.target.value))}
              className="accent-signal"
            />
            <span className="text-xs text-ink-soft">Moving to another frame re-detects the subject there.</span>
          </label>

          {error && <p className="text-sm font-medium text-signal">{error}</p>}

          <button
            type="button"
            onClick={onStartOver}
            className="mt-auto rounded-xl border border-steel px-4 py-2 text-sm font-semibold transition hover:bg-mist"
          >
            Start over
          </button>
        </aside>
      </section>

      <section>
        <h2 className="mb-4 text-2xl font-semibold">Choose a style</h2>
        <div className="grid gap-5 md:grid-cols-3">
          {styles.map((style) => {
            const selected = style.key === selectedStyle;
            return (
              <article
                key={style.key}
                className={
                  "flex flex-col overflow-hidden rounded-2xl bg-white transition " +
                  (selected ? "ring-3 ring-signal" : "ring-1 ring-steel")
                }
              >
                {subject ? (
                  <LoadingImage
                    src={stylePreviewUrl(project.id, style.key, subject, maxSide, look)}
                    alt={`${style.label} preview`}
                    className="border-b border-steel bg-white"
                    aspectRatio={`${width} / ${height}`}
                  />
                ) : (
                  <div className="animate-pulse bg-mist" style={{ aspectRatio: `${width} / ${height}` }} />
                )}
                <div className="flex flex-1 flex-col gap-3 p-4">
                  <div>
                    <h3 className="font-semibold">{style.label}</h3>
                    <p className="mt-1 text-sm text-ink-soft">{style.description}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => onSelectStyle(style.key)}
                    className={
                      "mt-auto rounded-xl px-4 py-2.5 font-semibold transition " +
                      (selected ? "bg-signal text-white" : "border border-steel hover:bg-mist")
                    }
                  >
                    {selected ? "Selected ✓" : "Choose this style"}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <SettingsPanel settings={settings} onChange={onSettingsChange} metadata={project.metadata} style={selectedStyle} />

      <div className="flex flex-col items-end gap-2">
        <button
          type="button"
          disabled={!selectedStyle || starting}
          onClick={async () => {
            setStarting(true);
            setGenerateError(null);
            try {
              await onGenerate();
            } catch (e) {
              setGenerateError((e as Error).message);
              setStarting(false);
            }
          }}
          className="rounded-xl bg-signal px-6 py-3 text-lg font-semibold text-white transition hover:brightness-90 disabled:cursor-not-allowed disabled:bg-steel disabled:text-ink-soft"
        >
          {starting ? "Starting…" : "Generate stroke video"}
        </button>
        {!selectedStyle && <p className="text-sm text-ink-soft">Choose a style first.</p>}
        {generateError && <p className="text-sm font-medium text-signal">{generateError}</p>}
      </div>
    </div>
  );
}
