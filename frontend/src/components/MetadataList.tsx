import type { VideoMetadata } from "../api";
import { formatBytes, formatDuration, formatFps } from "../lib/format";

export function MetadataList({ metadata }: { metadata: VideoMetadata }) {
  const rows: [string, string][] = [
    ["Resolution", `${metadata.width} × ${metadata.height}`],
    ["Frame rate", `${formatFps(metadata.fps)} fps`],
    ["Duration", formatDuration(metadata.duration_s)],
    ["File size", formatBytes(metadata.size_bytes)],
    ["Codec", metadata.codec + (metadata.has_audio ? " + audio" : "")],
  ];
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-ink-soft">{label}</dt>
          <dd className="font-medium tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
