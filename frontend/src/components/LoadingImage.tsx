import { useLayoutEffect, useRef, useState } from "react";

/** <img> with a pulsing placeholder while loading and a message if the server errors. */
export function LoadingImage({ src, alt, className = "", aspectRatio }: {
  src: string;
  alt: string;
  className?: string;
  aspectRatio?: string;
}) {
  const [state, setState] = useState<{ src: string; status: "loading" | "loaded" | "error" }>({
    src,
    status: "loading",
  });
  const img = useRef<HTMLImageElement>(null);
  // Reset when the URL changes (new frame, new subject) without remounting.
  const status = state.src === src ? state.status : "loading";

  // A cached image can finish loading before React attaches onLoad; catch that case.
  useLayoutEffect(() => {
    if (img.current?.complete && img.current.naturalWidth > 0) setState({ src, status: "loaded" });
  }, [src]);

  return (
    <div className={"relative overflow-hidden " + className} style={{ aspectRatio }}>
      <img
        ref={img}
        src={src}
        alt={alt}
        draggable={false}
        onLoad={() => setState({ src, status: "loaded" })}
        onError={() => setState({ src, status: "error" })}
        className={"block h-full w-full object-contain " + (status === "loaded" ? "" : "invisible")}
      />
      {status === "loading" && <div className="absolute inset-0 animate-pulse bg-mist" />}
      {status === "error" && (
        <div className="absolute inset-0 flex items-center justify-center bg-mist p-4 text-center text-sm text-ink-soft">
          Couldn't render this preview. Check the backend log.
        </div>
      )}
    </div>
  );
}
