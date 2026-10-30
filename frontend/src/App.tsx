import { useState } from "react";
import type { Project } from "./api";
import { ProjectReady } from "./components/ProjectReady";
import { StepHeader } from "./components/StepHeader";
import { UploadStep } from "./components/UploadStep";

export default function App() {
  const [project, setProject] = useState<Project | null>(null);

  return (
    <div className="min-h-screen">
      <StepHeader current={project ? "Style" : "Upload"} />
      <main className="mx-auto max-w-6xl px-4 py-8">
        {project ? (
          <ProjectReady project={project} onStartOver={() => setProject(null)} />
        ) : (
          <UploadStep onConfirmed={setProject} />
        )}
      </main>
    </div>
  );
}
