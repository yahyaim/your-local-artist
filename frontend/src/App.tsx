import { useCallback, useEffect, useState } from "react";
import { deleteProject, getProject, getSystem, startJob, type GenerateSettings, type Project, type SystemInfo } from "./api";
import { GenerateStep } from "./components/GenerateStep";
import { ResultStep } from "./components/ResultStep";
import { StepHeader, type Step } from "./components/StepHeader";
import { StyleStep } from "./components/StyleStep";
import { SystemBanner } from "./components/SystemBanner";
import { UploadStep } from "./components/UploadStep";

type Screen = { kind: "style" } | { kind: "generate"; jobId: string } | { kind: "result"; renderId: string };

const STEP_FOR: Record<Screen["kind"], Step> = { style: "Style", generate: "Generate", result: "Result" };

function defaultSettings(system: SystemInfo, project: Project): GenerateSettings {
  const { max_side, half_fps, max_seconds } = system.defaults;
  const duration = project.metadata.duration_s;
  return {
    thickness: 1,
    detail: 0.5,
    smoothing: 0.5,
    remove_background: true,
    paper_texture: true,
    max_side: Math.min(max_side, Math.max(project.metadata.width, project.metadata.height)),
    half_fps,
    start_s: 0,
    // On CPU, default to a short first clip.
    end_s: max_seconds && duration > max_seconds ? max_seconds : null,
    low_memory: false,
  };
}

/** The URL hash mirrors where you are (#project=…&job=… / &render=…) so a reload keeps your place. */
function writeHash(project: Project | null, screen: Screen) {
  const q = new URLSearchParams();
  if (project) {
    q.set("project", project.id);
    if (screen.kind === "generate") q.set("job", screen.jobId);
    if (screen.kind === "result") q.set("render", screen.renderId);
  }
  history.replaceState(null, "", q.size ? `#${q}` : " ");
}

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [screen, setScreen] = useState<Screen>({ kind: "style" });
  const [style, setStyle] = useState<string | null>(null);
  const [settings, setSettings] = useState<GenerateSettings | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [systemError, setSystemError] = useState<string | null>(null);

  // Poll while model weights are downloading (first run) so progress stays live.
  useEffect(() => {
    let timer: number | undefined;
    const load = () =>
      getSystem()
        .then((info) => {
          setSystem(info);
          setSystemError(null);
          if (info.models.some((m) => m.status === "downloading" || m.status === "missing")) {
            timer = window.setTimeout(load, 1500);
          }
        })
        .catch(() => {
          setSystemError("Can't reach the backend. Is it running (make dev)?");
          timer = window.setTimeout(load, 3000);
        });
    load();
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const q = new URLSearchParams(window.location.hash.slice(1));
    const id = q.get("project");
    if (!id) return;
    getProject(id)
      .then((p) => {
        setProject(p);
        const job = q.get("job");
        const render = q.get("render");
        if (job) setScreen({ kind: "generate", jobId: job });
        else if (render) setScreen({ kind: "result", renderId: render });
      })
      .catch(() => history.replaceState(null, "", " "));
  }, []);

  useEffect(() => writeHash(project, screen), [project, screen]);

  useEffect(() => {
    if (project && system && !settings) setSettings(defaultSettings(system, project));
  }, [project, system, settings]);

  const onDone = useCallback((renderId: string) => setScreen({ kind: "result", renderId }), []);
  const backToStyles = useCallback(() => setScreen({ kind: "style" }), []);

  function startOver() {
    // Free the disk space right away; anything left behind is swept after PROJECT_TTL_HOURS.
    if (project) deleteProject(project.id).catch(() => {});
    setProject(null);
    setStyle(null);
    setSettings(null);
    setScreen({ kind: "style" });
  }

  async function generate() {
    if (!project || !style || !settings) return;
    const job = await startJob(project.id, style, settings);
    setScreen({ kind: "generate", jobId: job.id });
  }

  return (
    <div className="min-h-screen">
      <StepHeader current={project ? STEP_FOR[screen.kind] : "Upload"} />
      <main className="mx-auto max-w-6xl px-4 py-8">
        {systemError && <p className="mb-6 rounded-xl bg-white p-3 text-sm font-medium text-signal">{systemError}</p>}
        {system && <SystemBanner system={system} />}
        {!project ? (
          <UploadStep onConfirmed={setProject} />
        ) : screen.kind === "generate" ? (
          <GenerateStep jobId={screen.jobId} onDone={onDone} onBack={backToStyles} />
        ) : screen.kind === "result" ? (
          system ? (
            <ResultStep project={project} renderId={screen.renderId} blender={system.blender}
                        onTryAnotherStyle={backToStyles} onStartOver={startOver} />
          ) : null
        ) : system && settings ? (
          <StyleStep
            project={project}
            settings={settings}
            onSettingsChange={setSettings}
            selectedStyle={style}
            onSelectStyle={setStyle}
            onGenerate={generate}
            onStartOver={startOver}
          />
        ) : null}
      </main>
    </div>
  );
}
