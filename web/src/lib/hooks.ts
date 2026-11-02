"use client";

import { useCallback, useEffect, useState } from "react";

import { getJSON } from "./api";

type PollState<T> = { data?: T; error?: string; updatedAt?: number };

/** Fetch `path` now and every `intervalMs`; skips ticks while the tab is hidden. */
export function usePoll<T>(path: string | null, intervalMs: number, paused = false) {
  const [state, setState] = useState<PollState<T>>({});
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let first = true;

    const schedule = () => {
      if (alive && !paused) timer = setTimeout(run, intervalMs);
    };
    const run = async () => {
      clearTimeout(timer);
      // Background tabs skip ticks, but never the first fetch: a page needs data once.
      if (document.hidden && !first) return schedule();
      first = false;
      controller?.abort();
      controller = new AbortController();
      try {
        const data = await getJSON<T>(path, { signal: controller.signal });
        if (alive) setState({ data, updatedAt: Date.now() });
      } catch (err) {
        if (alive && !(err instanceof DOMException && err.name === "AbortError")) {
          setState((s) => ({ ...s, error: err instanceof Error ? err.message : String(err) }));
        }
      }
      schedule();
    };
    const onVisible = () => {
      if (!document.hidden && !paused) run();
    };

    run();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [path, intervalMs, paused, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, refresh };
}

/** An object persisted to localStorage, read after mount so static HTML matches. */
export function useStored<T extends object>(key: string, initial: T) {
  const [value, setValue] = useState<T>(initial);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw != null) setValue({ ...initial, ...JSON.parse(raw) } as T);
    } catch {
      /* storage unavailable or corrupt: keep the default */
    }
    setReady(true);
  }, [key]); // initial is a fresh literal each render; key alone decides when to reload

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable */
    }
  }, [key, value, ready]);

  return [value, setValue, ready] as const;
}

/** Re-render every `ms`, for "updated 3s ago" style labels. */
export function useNow(ms = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}
