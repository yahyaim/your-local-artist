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

export const projectSourceUrl = (projectId: string) => `/api/projects/${projectId}/source`;
