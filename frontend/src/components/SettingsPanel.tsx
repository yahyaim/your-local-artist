import { useState } from "react";
import type { GenerateSettings, VideoMetadata } from "../api";
import { formatDuration } from "../lib/format";

interface Props {
  settings: GenerateSettings;
  onChange: (settings: GenerateSettings) => void;
  metadata: VideoMetadata;
  style: string | null;
}

function Slider({ label, value, min, max, step, format, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  format: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="flex justify-between">
        <span className="font-medium">{label}</span>
        <span className="tabular-nums text-ink-soft">{format(value)}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))} className="accent-signal" />
    </label>
  );
}

function Toggle({ label, hint, checked, onChange }: {
  label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 text-sm">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
             className="mt-0.5 h-4 w-4 accent-signal" />
      <span>
        <span className="font-medium">{label}</span>
        {hint && <span className="block text-xs text-ink-soft">{hint}</span>}
      </span>
    </label>
  );
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Collapsible settings with sensible defaults. Look settings also drive the style card previews. */
export function SettingsPanel({ settings, onChange, metadata, style }: Props) {
  const [open, setOpen] = useState(false);
  const set = <K extends keyof GenerateSettings>(key: K, value: GenerateSettings[K]) =>
    onChange({ ...settings, [key]: value });

  const longSide = Math.max(metadata.width, metadata.height);
  const sizes = [...new Set([360, 480, 720, 1080, 1440].filter((s) => s < longSide).concat(longSide))];
  const end = settings.end_s ?? metadata.duration_s;
  const fps = metadata.fps / (settings.half_fps ? 2 : 1);
  const frames = Math.max(0, Math.round((end - settings.start_s) * fps));

  return (
    <section className="rounded-2xl border border-steel bg-white">
      <button type="button" onClick={() => setOpen(!open)}
              className="flex w-full items-center justify-between px-5 py-4 text-left">
        <span>
          <span className="font-semibold">Settings</span>
          <span className="ml-3 text-sm text-ink-soft">
            {settings.max_side}px · {fps.toFixed(fps % 1 ? 2 : 0)} fps · {formatDuration(settings.start_s)}–
            {formatDuration(end)} ({frames} frames)
          </span>
        </span>
        <span className="text-ink-soft">{open ? "Hide ▲" : "Show ▼"}</span>
      </button>

      {open && (
        <div className="grid gap-x-8 gap-y-5 border-t border-steel px-5 py-5 md:grid-cols-2">
          <div className="flex flex-col gap-5">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-soft">Look</h3>
            <Slider label="Stroke thickness" value={settings.thickness} min={0.25} max={4} step={0.05}
                    format={(v) => `${v.toFixed(2)}×`} onChange={(v) => set("thickness", v)} />
            <Slider label="Line detail" value={settings.detail} min={0} max={1} step={0.05}
                    format={pct} onChange={(v) => set("detail", v)} />
            <Slider label="Smoothing (over time + along lines)" value={settings.smoothing} min={0} max={1} step={0.05}
                    format={pct} onChange={(v) => set("smoothing", v)} />
            <Toggle label="Remove background" hint="Draw only the subject (SAM 2 mask)."
                    checked={settings.remove_background} onChange={(v) => set("remove_background", v)} />
            {style === "pencil" && (
              <Toggle label="Paper texture" hint="Preview video only; not part of the vector strokes."
                      checked={settings.paper_texture} onChange={(v) => set("paper_texture", v)} />
            )}
          </div>

          <div className="flex flex-col gap-5">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-soft">Output</h3>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="font-medium">Resolution (long side)</span>
              <select value={settings.max_side} onChange={(e) => set("max_side", Number(e.target.value))}
                      className="rounded-lg border border-steel bg-white px-3 py-2">
                {sizes.map((s) => (
                  <option key={s} value={s}>{s}px{s === longSide ? " (source)" : ""}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="font-medium">Frame rate</span>
              <select value={settings.half_fps ? "half" : "source"}
                      onChange={(e) => set("half_fps", e.target.value === "half")}
                      className="rounded-lg border border-steel bg-white px-3 py-2">
                <option value="source">Source ({metadata.fps.toFixed(metadata.fps % 1 ? 2 : 0)} fps)</option>
                <option value="half">Half ({(metadata.fps / 2).toFixed(metadata.fps % 2 ? 2 : 0)} fps)</option>
              </select>
            </label>
            <Slider label="Start" value={settings.start_s} min={0} max={metadata.duration_s} step={0.1}
                    format={formatDuration}
                    onChange={(v) => onChange({ ...settings, start_s: Math.min(v, end - 0.1) })} />
            <Slider label="End" value={end} min={0} max={metadata.duration_s} step={0.1} format={formatDuration}
                    onChange={(v) => {
                      const clamped = Math.max(v, settings.start_s + 0.1);
                      set("end_s", clamped >= metadata.duration_s - 0.05 ? null : clamped);
                    }} />
            <Toggle label="Low memory mode" hint="Smaller SAM 2 model, smaller batches, models unloaded between stages. Slower; use if you run out of GPU memory."
                    checked={settings.low_memory} onChange={(v) => set("low_memory", v)} />
          </div>
        </div>
      )}
    </section>
  );
}
