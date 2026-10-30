import { projectSourceUrl, type Project } from "../api";
import { MetadataList } from "./MetadataList";

/** Placeholder after confirming; style selection replaces this in M2. */
export function ProjectReady({ project, onStartOver }: { project: Project; onStartOver: () => void }) {
  return (
    <section className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="overflow-hidden rounded-2xl bg-ink">
        <video src={projectSourceUrl(project.id)} controls playsInline className="max-h-[70vh] w-full" />
      </div>
      <aside className="flex flex-col gap-5 rounded-2xl border border-steel bg-white p-5">
        <div>
          <p className="truncate font-semibold" title={project.original_name}>{project.original_name}</p>
          <p className="text-xs text-ink-soft">Saved as project {project.id.slice(0, 8)}</p>
        </div>
        <MetadataList metadata={project.metadata} />
        <p className="rounded-xl bg-mist p-3 text-sm">Style selection comes in the next milestone (M2).</p>
        <button
          type="button"
          onClick={onStartOver}
          className="mt-auto rounded-xl border border-steel px-4 py-3 font-semibold transition hover:bg-mist"
        >
          Start over
        </button>
      </aside>
    </section>
  );
}
