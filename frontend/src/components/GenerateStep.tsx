import { useEffect, useState } from "react";
import { cancelJob, getJob, jobEventsUrl, type JobSnapshot } from "../api";

interface Props {
  jobId: string;
  onDone: (renderId: string) => void;
  onBack: () => void;
}

function formatEta(seconds: number): string {
  if (seconds < 60) return `about ${Math.max(5, Math.round(seconds / 5) * 5)} s left`;
  const m = Math.round(seconds / 60);
  return m < 60 ? `about ${m} min left` : `about ${Math.floor(m / 60)} h ${m % 60} min left`;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  return `${m}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

const STATUS_TEXT: Record<string, string> = { cached: "cached", skipped: "skipped", done: "done" };

/** Step 5: progress over Server-Sent Events (falls back to polling), with cancel. */
export function GenerateStep({ jobId, onDone, onBack }: Props) {
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [lost, setLost] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    let closed = false;
    let pollTimer: number | undefined;
    const handle = (snap: JobSnapshot) => {
      setJob(snap);
      if (snap.status === "done" && snap.render_id) onDone(snap.render_id);
      return ["done", "failed", "cancelled"].includes(snap.status);
    };
    const poll = () =>
      getJob(jobId)
        .then((snap) => {
          if (!closed && !handle(snap)) pollTimer = window.setTimeout(poll, 1000);
        })
        .catch((e: Error) => setLost(e.message));

    const events = new EventSource(jobEventsUrl(jobId));
    events.onmessage = (e) => {
      if (handle(JSON.parse(e.data))) events.close();
    };
    events.onerror = () => {
      // Stream dropped (proxy, sleep, backend restart): fall back to polling.
      events.close();
      if (!closed) poll();
    };
    return () => {
      closed = true;
      events.close();
      window.clearTimeout(pollTimer);
    };
  }, [jobId, onDone]);

  if (lost) {
    return (
      <section className="mx-auto max-w-2xl rounded-2xl border border-steel bg-white p-6">
        <p className="font-medium text-signal">{lost}</p>
        <button type="button" onClick={onBack} className="mt-4 rounded-xl border border-steel px-4 py-2 font-semibold hover:bg-mist">
          Back to styles
        </button>
      </section>
    );
  }
  if (!job) return <p className="text-ink-soft">Connecting…</p>;

  const current = job.stages.find((s) => s.key === job.stage);
  const finished = ["failed", "cancelled"].includes(job.status);

  return (
    <section className="mx-auto flex max-w-2xl flex-col gap-6 rounded-2xl border border-steel bg-white p-6">
      <div>
        <h2 className="text-2xl font-semibold">
          {job.status === "queued" && "Waiting for the previous job…"}
          {job.status === "running" && "Generating stroke video"}
          {job.status === "done" && "Done"}
          {job.status === "failed" && "Generation failed"}
          {job.status === "cancelled" && "Cancelled"}
        </h2>
        {job.status === "running" && (
          <p className="mt-1 text-sm text-ink-soft">
            {current ? `${current.label} · ${Math.round(current.progress * 100)}%` : "Starting…"}
            {" · "}
            {formatElapsed(job.elapsed_s)} elapsed
            {job.eta_s != null && job.progress > 0.01 && ` · ${formatEta(job.eta_s)}`}
          </p>
        )}
      </div>

      <div className="h-3 overflow-hidden rounded-full bg-mist" role="progressbar"
           aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-signal transition-[width] duration-300" style={{ width: `${job.progress * 100}%` }} />
      </div>

      <ol className="flex flex-col gap-2 text-sm">
        {job.stages.map((s) => (
          <li key={s.key} className="flex items-center gap-3">
            <span className={
              "flex h-5 w-5 items-center justify-center rounded-full text-xs " +
              (s.status === "done" || s.status === "cached" ? "bg-ink text-white"
                : s.status === "running" ? "animate-pulse bg-signal text-white"
                : "bg-mist text-ink-soft")
            }>
              {s.status === "done" || s.status === "cached" ? "✓" : s.status === "skipped" ? "–" : ""}
            </span>
            <span className={s.status === "pending" || s.status === "skipped" ? "text-ink-soft" : "font-medium"}>
              {s.label}
            </span>
            <span className="ml-auto tabular-nums text-ink-soft">
              {s.status === "running" ? `${Math.round(s.progress * 100)}%` : STATUS_TEXT[s.status] ?? ""}
            </span>
          </li>
        ))}
      </ol>

      {job.error && <p className="rounded-xl bg-mist p-3 text-sm font-medium text-signal">{job.error}</p>}

      <div className="flex justify-end gap-3">
        {finished ? (
          <button type="button" onClick={onBack}
                  className="rounded-xl border border-steel px-4 py-2 font-semibold hover:bg-mist">
            Back to styles
          </button>
        ) : (
          <button
            type="button"
            disabled={cancelling || job.status === "done"}
            onClick={() => {
              setCancelling(true);
              cancelJob(jobId).catch(() => setCancelling(false));
            }}
            className="rounded-xl border border-steel px-4 py-2 font-semibold hover:bg-mist disabled:opacity-50"
          >
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>
    </section>
  );
}
