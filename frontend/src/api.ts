// Typed client for the FastAPI backend. Vite proxies /api to it in dev.

export interface VideoMetadata {
  width: number;
  height: number;
  fps: number;
  duration_s: number;
  size_bytes: number;
  codec: string;
  has_audio: boolean;
}

export interface UploadResult {
  upload_id: string;
  filename: string;
  metadata: VideoMetadata;
}

export interface Project {
  id: string;
  created_at: string;
  original_name: string;
  source_file: string;
  metadata: VideoMetadata;
}

export class ApiError extends Error {}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through */
  }
  return `Request failed (${res.status})`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) throw new ApiError(await errorMessage(res));
  return (res.status === 204 ? undefined : await res.json()) as T;
}

/** Streams the file as the raw request body. XHR (not fetch) so we get upload progress. */
export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
): { done: Promise<UploadResult>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const done = new Promise<UploadResult>((resolve, reject) => {
    xhr.open("POST", `/api/uploads?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: { detail?: unknown } | UploadResult | null = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error */
      }
      if (xhr.status === 201 && body) resolve(body as UploadResult);
      else {
        const detail = body && "detail" in body && typeof body.detail === "string" ? body.detail : null;
        reject(new ApiError(detail ?? `Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new ApiError("Upload failed: is the backend running?"));
    xhr.onabort = () => reject(new DOMException("Upload aborted", "AbortError"));
    xhr.send(file);
  });
  return { done, abort: () => xhr.abort() };
}

export const discardUpload = (uploadId: string) =>
  request<void>(`/api/uploads/${uploadId}`, { method: "DELETE" });

export const confirmUpload = (uploadId: string) =>
  request<Project>(`/api/uploads/${uploadId}/confirm`, { method: "POST" });

export const getProject = (projectId: string) => request<Project>(`/api/projects/${projectId}`);

/** Deletes the project folder on the server: source video, renders and exports. */
export const deleteProject = (projectId: string) =>
  request<void>(`/api/projects/${projectId}`, { method: "DELETE" });

export const projectSourceUrl = (projectId: string) => `/api/projects/${projectId}/source`;

// --- M2: system, subject selection, style previews ---------------------------

export interface ModelStatus {
  key: string;
  label: string;
  status: "missing" | "downloading" | "ready" | "error";
  done_mb: number;
  total_mb: number;
  error: string | null;
}

export interface BlenderStatus {
  available: boolean;
  path: string | null;
  version: string | null;
  api: "GPv3" | "GPv2" | null;
  error: string | null;
}

export interface SystemInfo {
  device: { kind: "cuda" | "mps" | "cpu"; name: string; is_slow: boolean };
  blender: BlenderStatus;
  defaults: { max_side: number; half_fps: boolean; max_seconds: number | null };
  models: ModelStatus[];
}

export interface StyleInfo {
  key: string;
  label: string;
  description: string;
}

/** A click on the subject: normalised x, y in [0, 1], then 1 = include, 0 = exclude. */
export type SubjectPoint = [number, number, 0 | 1];

export interface Subject {
  time: number;
  mode: "auto" | "click";
  box: number[] | null;
  points: SubjectPoint[];
  version: string;
}

export const getSystem = () => request<SystemInfo>("/api/system");
export const getStyles = () => request<StyleInfo[]>("/api/styles");

export const getSubject = (projectId: string, t: number, maxSide: number) =>
  request<Subject>(`/api/projects/${projectId}/subject?t=${t}&max_side=${maxSide}`);

export const setSubject = (projectId: string, t: number, points: SubjectPoint[], maxSide: number) =>
  request<Subject>(`/api/projects/${projectId}/subject?max_side=${maxSide}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ time: t, points }),
  });

export const frameUrl = (projectId: string, t: number, maxSide: number) =>
  `/api/projects/${projectId}/frame?t=${t}&max_side=${maxSide}`;

// `v` (subject version) only busts the browser cache when the selection changes.
export const subjectOverlayUrl = (projectId: string, subject: Subject, maxSide: number) =>
  `/api/projects/${projectId}/subject/overlay?t=${subject.time}&max_side=${maxSide}&v=${subject.version}`;

/** Look settings shared by the style previews and generation (backend: StyleParams). */
export interface LookSettings {
  thickness: number;
  detail: number;
  smoothing: number;
  remove_background: boolean;
  paper_texture: boolean;
}

/** Everything the Settings panel controls (backend: GenerateSettings). */
export interface GenerateSettings extends LookSettings {
  max_side: number;
  half_fps: boolean;
  start_s: number;
  end_s: number | null;
  low_memory: boolean;
}

export const stylePreviewUrl = (
  projectId: string, style: string, subject: Subject, maxSide: number, look: LookSettings,
) => {
  const q = new URLSearchParams({
    t: String(subject.time),
    max_side: String(maxSide),
    v: subject.version,
    thickness: String(look.thickness),
    detail: String(look.detail),
    smoothing: String(look.smoothing),
    remove_background: String(look.remove_background),
    paper_texture: String(look.paper_texture),
  });
  return `/api/projects/${projectId}/styles/${style}/preview?${q}`;
};

// --- M3: generation jobs and renders -------------------------------------------

export type StageStatus = "pending" | "running" | "done" | "cached" | "skipped";

export interface JobSnapshot {
  id: string;
  project_id: string;
  style: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  render_id: string | null;
  stage: string | null;
  stages: { key: string; label: string; status: StageStatus; progress: number }[];
  progress: number;
  eta_s: number | null;
  elapsed_s: number;
  error: string | null;
}

export interface RenderInfo {
  id: string;
  style: string;
  settings: GenerateSettings;
  width: number;
  height: number;
  fps: number;
  frames: number;
  start_s: number;
  end_s: number;
  created_at: string;
  seconds_taken: number;
}

export const startJob = (projectId: string, style: string, settings: GenerateSettings) =>
  request<JobSnapshot>(`/api/projects/${projectId}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ style, settings }),
  });

export const getJob = (jobId: string) => request<JobSnapshot>(`/api/jobs/${jobId}`);
export const cancelJob = (jobId: string) => request<{ ok: boolean }>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
export const jobEventsUrl = (jobId: string) => `/api/jobs/${jobId}/events`;

export const getRender = (projectId: string, renderId: string) =>
  request<RenderInfo>(`/api/projects/${projectId}/renders/${renderId}`);

export const renderFileUrl = (projectId: string, renderId: string, name: "preview.mp4" | "original.mp4", download = false) =>
  `/api/projects/${projectId}/renders/${renderId}/${name}${download ? "?download=true" : ""}`;

// --- M5: exports ------------------------------------------------------------------

export type ExportKind = "svg" | "blend";

const exportPath = (projectId: string, renderId: string, kind: ExportKind) =>
  `/api/projects/${projectId}/renders/${renderId}/exports/${kind}`;

/** Builds the export on the server (cached after the first time). */
export const buildExport = (projectId: string, renderId: string, kind: ExportKind) =>
  request<{ ready: boolean; size_bytes: number }>(exportPath(projectId, renderId, kind), { method: "POST" });

export const exportDownloadUrl = exportPath;
