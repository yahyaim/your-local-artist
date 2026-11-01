import type { SystemInfo } from "../api";

/** CPU warning and first-run model download progress. Renders nothing when all is well. */
export function SystemBanner({ system }: { system: SystemInfo }) {
  const pending = system.models.filter((m) => m.status === "downloading" || m.status === "error");
  if (!system.device.is_slow && pending.length === 0) return null;

  return (
    <div className="mb-6 flex flex-col gap-2">
      {system.device.is_slow && (
        <p className="rounded-xl border border-signal/40 bg-white p-3 text-sm">
          <strong>No GPU found, running on CPU.</strong> Everything works but is much slower; previews default to{" "}
          {system.defaults.max_side}px. Try a short time range first.
        </p>
      )}
      {pending.map((m) => (
        <div key={m.key} className="rounded-xl bg-mist p-3 text-sm">
          {m.status === "error" ? (
            <span className="font-medium text-signal">
              Downloading {m.label} failed: {m.error}. Check your connection; it retries on the next preview.
            </span>
          ) : (
            <>
              <div className="mb-1 flex justify-between">
                <span>Downloading {m.label} (first run only)…</span>
                <span className="tabular-nums">
                  {m.done_mb.toFixed(0)} / {m.total_mb.toFixed(0)} MB
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white">
                <div className="h-full bg-ink" style={{ width: `${(100 * m.done_mb) / Math.max(m.total_mb, 1)}%` }} />
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}
