const STEPS = ["Upload", "Style", "Generate", "Result"] as const;
export type Step = (typeof STEPS)[number];

export function StepHeader({ current }: { current: Step }) {
  const currentIndex = STEPS.indexOf(current);
  return (
    <header className="border-b border-steel bg-white/60">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 py-4">
        <h1 className="text-lg font-bold tracking-tight">
          my-local-<span className="underline decoration-signal decoration-2 underline-offset-4">artist</span>
        </h1>
        <ol className="flex items-center gap-1 text-sm">
          {STEPS.map((step, i) => (
            <li key={step} className="flex items-center gap-1">
              <span
                className={
                  "rounded-full px-3 py-1 " +
                  (i === currentIndex
                    ? "bg-ink text-white"
                    : i < currentIndex
                      ? "bg-mist text-ink"
                      : "text-ink-soft")
                }
              >
                {i + 1}. {step}
              </span>
              {i < STEPS.length - 1 && <span className="text-steel">—</span>}
            </li>
          ))}
        </ol>
      </div>
    </header>
  );
}
