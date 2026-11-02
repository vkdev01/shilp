"use client";

// Charts drawn as plain SVG. Four chart types do not justify a charting
// library, and owning the drawing keeps gaps honest: a bucket with no traffic
// is a gap in a latency line, not a fall to zero.

import { useEffect, useId, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";

import { clock } from "@/lib/format";
import { cx } from "./ui";

type Numeric<T> = { [K in keyof T]: T[K] extends number ? K : never }[keyof T] & string;

export type ChartSeries<T> = {
  key: Numeric<T>;
  label: string;
  color: string;
  kind: "line" | "area" | "bar";
  /** Treat 0 as "no data" and break the line there. */
  gaps?: boolean;
  dashed?: boolean;
};

type Point = readonly [number, number];

function useWidth<E extends HTMLElement>() {
  const ref = useRef<E>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Axis maximum whose quarter is a round number, so every gridline label is clean. */
function niceMax(v: number): number {
  if (!(v > 0)) return 1;
  const step = v / 4;
  const exp = 10 ** Math.floor(Math.log10(step));
  const f = step / exp;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * exp * 4;
}

function segments(points: (Point | null)[]): Point[][] {
  const out: Point[][] = [];
  let current: Point[] = [];
  for (const p of points) {
    if (p) current.push(p);
    else if (current.length) {
      out.push(current);
      current = [];
    }
  }
  if (current.length) out.push(current);
  return out;
}

const path = (pts: Point[]) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");

export function TimeChart<T extends { ts: number }>({
  data,
  series,
  height = 190,
  format = (v) => v.toLocaleString(),
  stacked = false,
  empty = "No traffic in this window",
}: {
  data: T[];
  series: ChartSeries<T>[];
  height?: number;
  format?: (v: number) => string;
  stacked?: boolean;
  empty?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const uid = useId().replace(/:/g, "");

  const pad = { l: 46, r: 10, t: 12, b: 24 };
  const w = Math.max(0, width - pad.l - pad.r);
  const h = height - pad.t - pad.b;
  const n = data.length;
  const step = n ? w / n : 0;
  const val = (d: T, s: ChartSeries<T>) => Number(d[s.key]) || 0;

  const bars = series.filter((s) => s.kind === "bar");
  const lines = series.filter((s) => s.kind !== "bar");

  const max = useMemo(() => {
    let m = 0;
    for (const d of data) {
      const b = stacked
        ? bars.reduce((a, s) => a + val(d, s), 0)
        : Math.max(0, ...bars.map((s) => val(d, s)));
      m = Math.max(m, b, ...lines.map((s) => val(d, s)));
    }
    return niceMax(m * 1.08);
  }, [data, series, stacked]);

  const x = (i: number) => pad.l + step * (i + 0.5);
  const y = (v: number) => pad.t + h - (v / max) * h;
  const hasData = data.some((d) => series.some((s) => val(d, s) > 0));
  const bucket = n > 1 ? data[1].ts - data[0].ts : 60;
  const xTicks = n ? [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * (n - 1))) : [];

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const i = Math.floor((e.clientX - rect.left - pad.l) / step);
    setHover(i >= 0 && i < n ? i : null);
  };

  const tip = hover != null ? data[hover] : null;
  const tipLeft = hover != null ? Math.min(Math.max(x(hover), 90), width - 90) : 0;

  return (
    <div ref={ref} className="relative w-full select-none" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} onMouseMove={onMove} onMouseLeave={() => setHover(null)} className="block">
          <defs>
            {series.map((s) => (
              <linearGradient key={s.key} id={`${uid}-${s.key}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={0.28} />
                <stop offset="100%" stopColor={s.color} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>

          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line
                x1={pad.l}
                x2={pad.l + w}
                y1={y(max * f)}
                y2={y(max * f)}
                stroke="var(--line)"
                strokeDasharray={f === 0 ? undefined : "2 4"}
              />
              <text x={pad.l - 8} y={y(max * f)} dy="0.32em" textAnchor="end" className="fill-faint text-[10px] tabular">
                {format(max * f)}
              </text>
            </g>
          ))}
          {xTicks.map((i, k) => (
            <text
              key={k}
              x={x(i)}
              y={height - 6}
              textAnchor={k === 0 ? "start" : k === xTicks.length - 1 ? "end" : "middle"}
              className="fill-faint text-[10px] tabular"
            >
              {clock(data[i].ts, bucket < 60)}
            </text>
          ))}

          {hover != null && (
            <rect x={pad.l + step * hover} y={pad.t} width={step} height={h} fill="var(--subtle)" opacity={0.9} />
          )}

          {bars.length > 0 &&
            data.map((d, i) => {
              let base = 0;
              const bw = Math.max(1, step * 0.68);
              return bars.map((s, k) => {
                const v = val(d, s);
                if (v <= 0) return null;
                const from = stacked ? base : 0;
                base += v;
                const offset = stacked ? 0 : (k - (bars.length - 1) / 2) * (bw / bars.length);
                const width_ = stacked ? bw : bw / bars.length;
                return (
                  <rect
                    key={`${i}-${s.key}`}
                    x={x(i) - width_ / 2 + offset}
                    y={y(from + v)}
                    width={width_}
                    height={Math.max(1, y(from) - y(from + v))}
                    rx={Math.min(2, width_ / 3)}
                    fill={s.color}
                    opacity={hover == null || hover === i ? 0.9 : 0.55}
                  />
                );
              });
            })}

          {lines.map((s) => {
            const pts = data.map((d, i) => {
              const v = val(d, s);
              return s.gaps && v <= 0 ? null : ([x(i), y(v)] as const);
            });
            return (
              <g key={s.key}>
                {segments(pts).map((seg, k) =>
                  seg.length === 1 ? (
                    <circle key={k} cx={seg[0][0]} cy={seg[0][1]} r={2.5} fill={s.color} />
                  ) : (
                    <g key={k}>
                      {s.kind === "area" && (
                        <path
                          d={`${path(seg)}L${seg[seg.length - 1][0]},${y(0)}L${seg[0][0]},${y(0)}Z`}
                          fill={`url(#${uid}-${s.key})`}
                        />
                      )}
                      <path
                        d={path(seg)}
                        fill="none"
                        stroke={s.color}
                        strokeWidth={1.75}
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        strokeDasharray={s.dashed ? "4 3" : undefined}
                      />
                    </g>
                  ),
                )}
                {hover != null && pts[hover] && (
                  <circle cx={pts[hover]![0]} cy={pts[hover]![1]} r={3.5} fill="var(--panel)" stroke={s.color} strokeWidth={2} />
                )}
              </g>
            );
          })}
        </svg>
      )}

      {width > 0 && !hasData && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center pl-10 text-xs text-faint">
          {empty}
        </div>
      )}

      {tip && (
        <div
          className="pointer-events-none absolute top-1 z-10 min-w-36 -translate-x-1/2 rounded-lg border border-line bg-panel/95 px-3 py-2 text-xs shadow-card backdrop-blur"
          style={{ left: tipLeft }}
        >
          <div className="mb-1 font-medium text-muted tabular">
            {clock(tip.ts, bucket < 60)} – {clock(tip.ts + bucket, bucket < 60)}
          </div>
          {series.map((s) => {
            const v = val(tip, s);
            return (
              <div key={s.key} className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: s.color }} />
                  {s.label}
                </span>
                <span className="font-semibold tabular">{s.gaps && v <= 0 ? "—" : format(v)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <span
            className="h-0.5 w-3 rounded-full"
            style={{
              background: i.dashed ? `repeating-linear-gradient(90deg, ${i.color} 0 3px, transparent 3px 5px)` : i.color,
            }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}

/** A tiny trend line for KPI cards; zeros are gaps when `gaps` is set. */
export function Sparkline({
  values,
  color,
  kind = "area",
  gaps = false,
  height = 34,
}: {
  values: number[];
  color: string;
  kind?: "area" | "bar";
  gaps?: boolean;
  height?: number;
}) {
  const uid = useId().replace(/:/g, "");
  const n = values.length;
  const max = Math.max(...values, 0) || 1;
  const W = 100;
  const x = (i: number) => (n <= 1 ? W / 2 : (i / (n - 1)) * W);
  const y = (v: number) => height - 2 - (v / max) * (height - 4);

  if (kind === "bar") {
    const bw = W / Math.max(1, n);
    return (
      <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" className="h-full w-full" aria-hidden>
        {values.map((v, i) =>
          v > 0 ? (
            <rect key={i} x={i * bw + bw * 0.15} width={bw * 0.7} y={y(v)} height={height - 2 - y(v)} fill={color} opacity={0.75} />
          ) : null,
        )}
      </svg>
    );
  }

  const segs = segments(values.map((v, i) => (gaps && v <= 0 ? null : ([x(i), y(v)] as const))));
  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" className="h-full w-full overflow-visible" aria-hidden>
      <defs>
        <linearGradient id={uid} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      {segs.map((seg, k) =>
        seg.length === 1 ? (
          <line key={k} x1={seg[0][0]} x2={seg[0][0]} y1={seg[0][1]} y2={height - 2} stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        ) : (
          <g key={k}>
            <path d={`${path(seg)}L${seg[seg.length - 1][0]},${height}L${seg[0][0]},${height}Z`} fill={`url(#${uid})`} />
            <path d={path(seg)} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
          </g>
        ),
      )}
    </svg>
  );
}

/** Horizontal proportional bars, for breakdowns by model, surface or status. */
export function BarList({
  rows,
  format = (v) => v.toLocaleString(),
}: {
  rows: { label: ReactNode; value: number; color?: string; hint?: ReactNode }[];
  format?: (v: number) => string;
}) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <ul className="space-y-2">
      {rows.map((r, i) => (
        <li key={i} className="text-sm">
          <div className="mb-1 flex items-center justify-between gap-3">
            <span className="min-w-0 truncate">{r.label}</span>
            <span className="shrink-0 tabular text-muted">
              {r.hint && <span className="mr-2 text-xs text-faint">{r.hint}</span>}
              <span className="font-medium text-fg">{format(r.value)}</span>
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-subtle">
            <div
              className="h-full rounded-full transition-[width] duration-500"
              style={{ width: `${(r.value / max) * 100}%`, background: r.color ?? "var(--accent)" }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Distribution of values in `bins` equal-width buckets. */
export function Histogram({
  values,
  bins = 24,
  color = "var(--accent)",
  format = (v) => v.toFixed(0),
  height = 120,
  markers = [],
}: {
  values: number[];
  bins?: number;
  color?: string;
  format?: (v: number) => string;
  height?: number;
  markers?: { at: number; label: string }[];
}) {
  const [hover, setHover] = useState<number | null>(null);
  if (!values.length) return <div style={{ height }} className="flex items-center justify-center text-xs text-faint">No data</div>;
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[0];
  // Clip at p99 so a single outlier cannot squash the body into one bar.
  const hi = sorted[Math.max(0, Math.ceil(0.99 * sorted.length) - 1)];
  const clipped = hi < sorted[sorted.length - 1];
  const span = hi - lo || 1;
  const counts = new Array(bins).fill(0) as number[];
  for (const v of values) counts[Math.max(0, Math.min(bins - 1, Math.floor(((v - lo) / span) * bins)))] += 1;
  const peak = Math.max(...counts);
  const edge = (i: number) => lo + (span * i) / bins;

  return (
    <div>
      <div className="relative flex items-end gap-px" style={{ height }} onMouseLeave={() => setHover(null)}>
        {counts.map((c, i) => (
          <div key={i} className="group relative h-full flex-1" onMouseEnter={() => setHover(i)}>
            <div
              className={cx("absolute inset-x-0 bottom-0 rounded-t-sm transition-opacity", hover != null && hover !== i && "opacity-50")}
              style={{ height: `${(c / peak) * 100}%`, minHeight: c ? 2 : 0, background: color }}
            />
          </div>
        ))}
        {markers.map((m, i) => (
          <div
            key={m.label}
            className="pointer-events-none absolute inset-y-0 border-l border-dashed border-fg/40"
            style={{ left: `${Math.min(100, ((m.at - lo) / span) * 100)}%` }}
          >
            <span
              className="absolute left-1 rounded bg-panel/90 px-1 text-[10px] font-medium whitespace-nowrap text-muted"
              style={{ top: i * 16 }}
            >
              {m.label} {format(m.at)}
            </span>
          </div>
        ))}
        {hover != null && (
          <div className="pointer-events-none absolute -top-8 left-1/2 -translate-x-1/2 rounded-md border border-line bg-panel px-2 py-1 text-xs shadow-card tabular">
            {format(edge(hover))}–{format(edge(hover + 1))}: <b>{counts[hover]}</b>
          </div>
        )}
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] text-faint tabular">
        <span>{format(lo)}</span>
        <span>{clipped ? `≥ ${format(hi)} (p99)` : format(hi)}</span>
      </div>
    </div>
  );
}
