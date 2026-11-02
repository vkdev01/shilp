"use client";

import {
  Activity,
  AlertTriangle,
  Clock,
  Cpu,
  Gauge,
  HardDrive,
  Layers,
  Pause,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Timer,
  Zap,
} from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import type { Dist, Stats } from "@/lib/api";
import { bytes, compact, duration, ms, num, pct, rate } from "@/lib/format";
import { useNow, usePoll, useStored } from "@/lib/hooks";
import { BarList, Legend, Sparkline, TimeChart } from "./charts";
import { RequestsTable } from "./requests-table";
import { Badge, Button, Card, CardHeader, Code, Empty, Kv, PageHeader, Segmented, Skeleton, cx, statusTone } from "./ui";

const WINDOWS = [
  { value: 300, label: "5m" },
  { value: 900, label: "15m" },
  { value: 3600, label: "1h" },
  { value: 21600, label: "6h" },
  { value: 86400, label: "24h" },
];

const C = {
  accent: "var(--accent)",
  info: "var(--info)",
  violet: "var(--violet)",
  ok: "var(--ok)",
  bad: "var(--bad)",
  warn: "var(--warn)",
  teal: "var(--teal)",
};

function Kpi({
  icon,
  label,
  value,
  unit,
  sub,
  spark,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  unit?: string;
  sub: ReactNode;
  spark: ReactNode;
  tone?: "bad";
}) {
  return (
    <Card className="relative overflow-hidden p-4">
      <div className="flex items-center gap-2 text-xs font-medium text-muted">
        <span className={cx("grid size-6 place-items-center rounded-md bg-subtle", tone === "bad" ? "text-bad" : "text-accent")}>
          {icon}
        </span>
        {label}
      </div>
      <div className="mt-3 flex items-baseline gap-1">
        <span className={cx("text-2xl font-semibold tracking-tight tabular", tone === "bad" && "text-bad")}>{value}</span>
        {unit && <span className="text-sm text-muted">{unit}</span>}
      </div>
      <div className="mt-0.5 truncate text-xs text-muted tabular">{sub}</div>
      <div className="mt-3 h-9">{spark}</div>
    </Card>
  );
}

function Pill({ icon, label, value, hint }: { icon: ReactNode; label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex min-w-0 items-center gap-3 px-4 py-3">
      <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-subtle text-muted">{icon}</span>
      <div className="min-w-0">
        <div className="text-[11px] font-medium uppercase tracking-wider text-faint">{label}</div>
        <div className="truncate text-sm font-semibold tabular">{value}</div>
        {hint && <div className="truncate text-xs text-muted">{hint}</div>}
      </div>
    </div>
  );
}

function DistTable({ rows }: { rows: { label: string; dist: Dist; fmt: (v: number) => string; note: string }[] }) {
  const cols: (keyof Dist)[] = ["avg", "p50", "p90", "p95", "p99", "max"];
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-y border-line bg-subtle/50 text-[11px] font-semibold uppercase tracking-wider text-faint">
            <th className="px-5 py-2 text-left">Metric</th>
            {cols.map((c) => (
              <th key={c} className="px-3 py-2 text-right last:pr-5">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r.label}>
              <td className="px-5 py-3">
                <div className="font-medium">{r.label}</div>
                <div className="text-xs text-muted">{r.note}</div>
              </td>
              {cols.map((c) => (
                <td key={c} className={cx("px-3 py-3 text-right tabular last:pr-5", c === "p95" ? "font-semibold" : "text-muted")}>
                  {r.fmt(r.dist[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Loading() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-16" />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 6 }, (_, i) => (
          <Skeleton key={i} className="h-36 rounded-2xl" />
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Skeleton className="h-64 rounded-2xl" />
        <Skeleton className="h-64 rounded-2xl" />
      </div>
    </div>
  );
}

export function Overview() {
  const [prefs, setPrefs] = useStored("shilp-overview", { window: 3600, live: true });
  const { data, error, updatedAt, refresh } = usePoll<Stats>(
    `/api/stats?window=${prefs.window}&recent=8`,
    2000,
    !prefs.live,
  );
  const now = useNow(1000);
  const w = data?.window;
  const series = w?.series ?? [];
  const traffic = series.map((b) => ({ ts: b.ts, ok: b.requests - b.errors, errors: b.errors }));

  return (
    <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      <PageHeader
        title="Overview"
        subtitle="Live serving telemetry — every number comes from /api/stats, recorded per request in SQLite."
      >
        <Segmented
          label="Time window"
          size="sm"
          value={prefs.window}
          options={WINDOWS}
          onChange={(v) => setPrefs((p) => ({ ...p, window: v }))}
        />
        <Button size="sm" onClick={() => setPrefs((p) => ({ ...p, live: !p.live }))} aria-pressed={prefs.live}>
          {prefs.live ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          {prefs.live ? "Live" : "Paused"}
          {prefs.live && <span className="live-dot ml-0.5 size-1.5 rounded-full bg-ok text-ok" />}
        </Button>
        <Button size="sm" variant="quiet" onClick={refresh} title="Refresh now">
          <RefreshCw className="size-3.5" />
          {updatedAt ? `${Math.max(0, Math.round((now - updatedAt) / 1000))}s` : "—"}
        </Button>
      </PageHeader>

      {error && (
        <div className="flex items-center gap-2 rounded-xl border border-bad/30 bg-bad/10 px-4 py-3 text-sm text-bad">
          <AlertTriangle className="size-4" />
          Could not reach the server: {error}
        </div>
      )}

      {!data || !w ? (
        <Loading />
      ) : (
        <>
          <Card className="grid divide-y divide-line sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-5 lg:divide-x">
            <Pill
              icon={<Server className="size-4" />}
              label="Uptime"
              value={duration(data.server.uptime_s + (updatedAt ? (now - updatedAt) / 1000 : 0))}
              hint={`v${data.server.version} · ${data.server.engine}`}
            />
            <Pill
              icon={<Activity className="size-4" />}
              label="In flight"
              value={
                <span className="flex items-center gap-2">
                  {data.in_flight} {data.in_flight === 1 ? "stream" : "streams"}
                  {data.in_flight > 0 && <span className="live-dot size-1.5 rounded-full bg-accent text-accent" />}
                </span>
              }
              hint="requests holding a model"
            />
            <Pill
              icon={<Layers className="size-4" />}
              label="Resident"
              value={`${data.resident.length} / ${data.server.max_resident}`}
              hint={data.resident.map((r) => r.ref).join(", ") || "nothing loaded"}
            />
            <Pill
              icon={<HardDrive className="size-4" />}
              label="Store"
              value={`${data.store.models} models`}
              hint={`${bytes(data.store.bytes)} · ${data.store.blobs} blobs`}
            />
            <Pill
              icon={<Sparkles className="size-4" />}
              label="Lifetime"
              value={`${compact(data.lifetime.requests)} requests`}
              hint={`${compact(data.lifetime.completion_tokens)} tokens generated`}
            />
          </Card>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            <Kpi
              icon={<Activity className="size-3.5" />}
              label="Requests"
              value={num(w.requests)}
              sub={`${w.rps > 0 && w.rps < 0.01 ? "<0.01" : w.rps.toFixed(2)} req/s · ${w.cancelled} cancelled`}
              spark={<Sparkline kind="bar" values={series.map((b) => b.requests)} color={C.accent} />}
            />
            <Kpi
              icon={<Zap className="size-3.5" />}
              label="Decode speed (p50)"
              value={rate(w.tok_per_s.p50)}
              unit="tok/s"
              sub={`p95 ${rate(w.tok_per_s.p95)} · best ${rate(w.tok_per_s.max)} tok/s`}
              spark={<Sparkline gaps values={series.map((b) => b.tok_per_s)} color={C.teal} />}
            />
            <Kpi
              icon={<Timer className="size-3.5" />}
              label="Time to first token (p50)"
              value={ms(w.ttft_ms.p50)}
              sub={`p95 ${ms(w.ttft_ms.p95)} · p99 ${ms(w.ttft_ms.p99)}`}
              spark={<Sparkline gaps values={series.map((b) => b.ttft_p50_ms)} color={C.violet} />}
            />
            <Kpi
              icon={<Clock className="size-3.5" />}
              label="End-to-end latency (p95)"
              value={ms(w.latency_ms.p95)}
              sub={`p50 ${ms(w.latency_ms.p50)} · max ${ms(w.latency_ms.max)}`}
              spark={<Sparkline gaps values={series.map((b) => b.latency_p95_ms)} color={C.info} />}
            />
            <Kpi
              icon={<Cpu className="size-3.5" />}
              label="Tokens generated"
              value={compact(w.completion_tokens)}
              sub={`${compact(w.prompt_tokens)} prompt · ${rate(w.output_tok_per_s)} tok/s sustained`}
              spark={<Sparkline kind="bar" values={series.map((b) => b.tokens)} color={C.ok} />}
            />
            <Kpi
              icon={<AlertTriangle className="size-3.5" />}
              label="Error rate"
              value={pct(w.error_rate)}
              sub={`${w.errors} failed of ${w.requests}`}
              tone={w.errors ? "bad" : undefined}
              spark={<Sparkline kind="bar" values={series.map((b) => b.errors)} color={C.bad} />}
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader
                title="Traffic"
                subtitle={`Requests per ${duration(w.bucket_seconds)} bucket`}
                action={<Legend items={[{ label: "ok", color: C.accent }, { label: "errors", color: C.bad }]} />}
              />
              <div className="px-3 pb-3">
                <TimeChart
                  data={traffic}
                  stacked
                  series={[
                    { key: "ok", label: "Succeeded", color: C.accent, kind: "bar" },
                    { key: "errors", label: "Failed", color: C.bad, kind: "bar" },
                  ]}
                  format={(v) => compact(v)}
                />
              </div>
            </Card>
            <Card>
              <CardHeader
                title="Latency"
                subtitle="Per bucket, successful requests only"
                action={
                  <Legend
                    items={[
                      { label: "TTFT p50", color: C.violet },
                      { label: "total p50", color: C.info },
                      { label: "total p95", color: C.info, dashed: true },
                    ]}
                  />
                }
              />
              <div className="px-3 pb-3">
                <TimeChart
                  data={series}
                  series={[
                    { key: "ttft_p50_ms", label: "TTFT p50", color: C.violet, kind: "area", gaps: true },
                    { key: "latency_p50_ms", label: "Total p50", color: C.info, kind: "line", gaps: true },
                    { key: "latency_p95_ms", label: "Total p95", color: C.info, kind: "line", gaps: true, dashed: true },
                  ]}
                  format={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`)}
                />
              </div>
            </Card>
            <Card>
              <CardHeader title="Decode throughput" subtitle="Mean completion tokens per second of generation" />
              <div className="px-3 pb-3">
                <TimeChart
                  data={series}
                  series={[{ key: "tok_per_s", label: "tok/s", color: C.teal, kind: "area", gaps: true }]}
                  format={(v) => v.toFixed(v < 10 ? 1 : 0)}
                />
              </div>
            </Card>
            <Card>
              <CardHeader title="Tokens generated" subtitle="Completion tokens per bucket" />
              <div className="px-3 pb-3">
                <TimeChart
                  data={series}
                  series={[{ key: "tokens", label: "Tokens", color: C.ok, kind: "bar" }]}
                  format={(v) => compact(v)}
                />
              </div>
            </Card>
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <Card className="xl:col-span-2">
              <CardHeader
                icon={<Gauge className="size-4" />}
                title="Latency distribution"
                subtitle={`${num(w.requests - w.errors - w.cancelled)} successful requests in the window`}
              />
              <DistTable
                rows={[
                  { label: "Time to first token", dist: w.ttft_ms, fmt: ms, note: "load + prompt processing; what users feel" },
                  { label: "End-to-end latency", dist: w.latency_ms, fmt: ms, note: "request in → last token out" },
                  {
                    label: "Decode speed",
                    dist: w.tok_per_s,
                    fmt: (v) => (v > 0 ? `${rate(v)} tok/s` : "—"),
                    note: "output tokens ÷ generation time",
                  },
                ]}
              />
            </Card>
            <Card>
              <CardHeader title="Responses" subtitle="Status codes and API surfaces" />
              <div className="space-y-5 px-5 pb-5">
                {w.requests === 0 ? (
                  <p className="text-sm text-muted">No requests yet.</p>
                ) : (
                  <>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(w.by_status)
                        .sort()
                        .map(([code, count]) => (
                          <Badge key={code} tone={statusTone(Number(code))} className="px-2 py-1 text-xs">
                            {code} <span className="opacity-70">× {count}</span>
                          </Badge>
                        ))}
                    </div>
                    <BarList
                      rows={w.by_surface.map((g) => ({
                        label: (
                          <span className="flex items-center gap-2">
                            <Badge tone={g.surface === "openai" ? "info" : "violet"}>{g.surface}</Badge>
                            <span className="text-xs text-muted">{rate(g.tok_per_s)} tok/s</span>
                          </span>
                        ),
                        value: g.requests,
                        color: g.surface === "openai" ? C.info : C.violet,
                      }))}
                    />
                    <BarList
                      rows={w.by_endpoint.map((g) => ({
                        label: <span className="font-mono text-xs">{g.endpoint}</span>,
                        value: g.requests,
                        color: "var(--line-strong)",
                        hint: g.errors ? `${g.errors} err` : undefined,
                      }))}
                    />
                  </>
                )}
              </div>
            </Card>
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <Card className="xl:col-span-2">
              <CardHeader title="By model" subtitle="Window aggregates per model" />
              {w.by_model.length === 0 ? (
                <Empty title="No model has served a request in this window">
                  Send one from the <Link className="text-accent hover:underline" href="/chat/">Playground</Link>.
                </Empty>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[560px] text-sm">
                    <thead>
                      <tr className="border-y border-line bg-subtle/50 text-[11px] font-semibold uppercase tracking-wider text-faint">
                        <th className="px-5 py-2 text-left">Model</th>
                        <th className="px-3 py-2 text-right">Requests</th>
                        <th className="px-3 py-2 text-right">Errors</th>
                        <th className="px-3 py-2 text-right">Tokens</th>
                        <th className="px-3 py-2 text-right">TTFT p50</th>
                        <th className="px-3 py-2 text-right">Total p95</th>
                        <th className="px-5 py-2 text-right">tok/s</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {w.by_model.map((g) => (
                        <tr key={g.model} className="hover:bg-subtle/50">
                          <td className="px-5 py-2.5 font-mono text-xs">{g.model}</td>
                          <td className="px-3 py-2.5 text-right tabular">{num(g.requests)}</td>
                          <td className={cx("px-3 py-2.5 text-right tabular", g.errors ? "text-bad" : "text-muted")}>{g.errors}</td>
                          <td className="px-3 py-2.5 text-right tabular">{compact(g.completion_tokens)}</td>
                          <td className="px-3 py-2.5 text-right tabular">{ms(g.ttft_p50_ms)}</td>
                          <td className="px-3 py-2.5 text-right tabular">{ms(g.latency_p95_ms)}</td>
                          <td className="px-5 py-2.5 text-right font-medium tabular">{rate(g.tok_per_s)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
            <Card>
              <CardHeader title="Resident models" subtitle={`Up to ${data.server.max_resident} in memory (SHILP_MAX_RESIDENT)`} />
              <div className="space-y-2 px-5 pb-5">
                {data.resident.length === 0 ? (
                  <p className="text-sm text-muted">Nothing loaded. The first request loads its model.</p>
                ) : (
                  data.resident.map((r) => (
                    <div key={r.ref} className="rounded-xl border border-line bg-subtle/40 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-xs font-medium">{r.ref}</span>
                        {r.refs > 0 ? <Badge tone="accent">{r.refs} streaming</Badge> : <Badge>idle</Badge>}
                      </div>
                      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <div className="text-faint">Size</div>
                          <div className="font-medium tabular">{bytes(r.size)}</div>
                        </div>
                        <div>
                          <div className="text-faint">Loaded</div>
                          <div className="font-medium tabular">{duration(r.loaded_seconds)}</div>
                        </div>
                        <div>
                          <div className="text-faint">Idle</div>
                          <div className="font-medium tabular">{r.refs > 0 ? "—" : duration(r.idle_seconds)}</div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <Card className="xl:col-span-2">
              <CardHeader
                title="Recent requests"
                subtitle="Newest first"
                action={
                  <Link href="/requests/" className="text-xs font-medium text-accent hover:underline">
                    All requests →
                  </Link>
                }
              />
              {data.recent.length ? (
                <RequestsTable rows={data.recent} now={now} compact />
              ) : (
                <Empty icon={<Activity className="size-5" />} title="No requests recorded yet">
                  Try <Code>curl localhost:{data.server.port}/api/chat</Code> or the Playground.
                </Empty>
              )}
            </Card>
            <Card>
              <CardHeader icon={<Server className="size-4" />} title="Server" />
              <div className="px-5 pb-5">
                <Kv
                  items={[
                    ["Version", `v${data.server.version}`],
                    ["Engine", <Code key="e">{data.server.engine}</Code>],
                    ["Platform", data.server.platform],
                    ["Python", data.server.python],
                    ["Threads", data.server.threads ?? "auto"],
                    ["GPU layers", data.server.gpu_layers === -1 ? "all" : data.server.gpu_layers],
                    ["Default model", data.server.default_model ?? "—"],
                    ["Listening", `${data.server.host}:${data.server.port}`],
                    ["Lifetime avg TTFT", ms(data.lifetime.avg_ttft_ms)],
                    ["Lifetime best", `${rate(data.lifetime.best_tok_per_s)} tok/s`],
                  ]}
                />
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
