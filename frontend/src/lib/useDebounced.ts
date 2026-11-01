import { useEffect, useState } from "react";

/** `value`, but only after it has stopped changing for `ms`. Compared by JSON so object literals work. */
export function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  const key = JSON.stringify(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), ms);
    return () => window.clearTimeout(timer);
  }, [key, ms]);
  return debounced;
}
