"use client";

import { Download, ListTree, Pause, Play, Search } from "lucide-react";
import { useMemo, useState } from "react";

import type { RequestRow, Stats } from "@/lib/api";
import { ms, num, rate } from "@/lib/format";
import { useNow, usePoll } from "@/lib/hooks";
import { Histogram } from "./charts";
import { RequestsTable } from "./requests-table";
import { Button, Card, CardHeader, Empty, PageHeader, Segmented, Skeleton } from "./ui";

type Outcome = "all" | "ok" | "error";
type Metric = "ttft_ms" | "total_ms" | "tok_per_s";

const METRICS: { value: Metric; label: string; fmt: (v: number) => string; color: string }[] = [
  { value: "ttft_ms", label: "TTFT", fmt: (v) => ms(v), color: "var(--violet)" },
  { value: "total_ms", label: "Total", fmt: (v) => ms(v), color: "var(--info)" },
  { value: "tok_per_s", label: "tok/s", fmt: (v) => rate(v), color: "var(--teal)" },
];

function percentile(sorted: number[], p: number) {
  if (!sorted.length) return 0;
  return sorted[Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)];
}

function cell(v: unknown): string {
  // Neutralise spreadsheet formulas: a model name is user-controlled text.
  const s = typeof v === "string" && /^[=+\-@\t\r]/.test(v) ? `'${v}` : v;
  return JSON.stringify(s);
}

function csv(rows: RequestRow[]): string {
  const keys = Object.keys(rows[0] ?? {}) as (keyof RequestRow)[];
  return [keys.join(","), ...rows.map((r) => keys.map((k) => cell(r[k])).join(","))].join("\n");
}

export function Requests() {
  const [live, setLive] = useState(true);
  const { data, error } = usePoll<Stats>("/api/stats?window=60&recent=500", 3000, !live);
  const now = useNow(5000);

  const [query, setQuery] = useState("");
  const [outcome, setOutcome] = useState<Outcome>("all");
  const [surface, setSurface] = useState<"all" | "ollama" | "openai">("all");
  const [metric, setMetric] = useState<Metric>("ttft_ms");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (data?.recent ?? []).filter(
      (r) =>
        (outcome === "all" || (outcome === "ok" ? r.status < 400 : r.status >= 400)) &&
        (surface === "all" || r.surface === surface) &&
        (!q || r.model.toLowerCase().includes(q) || r.endpoint.includes(q) || String(r.status) === q),
    );
  }, [data, query, outcome, surface]);

  const values = useMemo(
    () =>
      rows
        .filter((r) => r.status < 400 && r[metric] > 0)
        .map((r) => r[metric])
        .sort((a, b) => a - b),
    [rows, metric],
  );
  const m = METRICS.find((x) => x.value === metric)!;
  const p50 = percentile(values, 50);
  const p95 = percentile(values, 95);

  const download = () => {
    const url = URL.createObjectURL(new Blob([csv(rows)], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "shilp-requests.csv" });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      <PageHeader title="Requests" subtitle="The newest 500 requests from the telemetry table, one row per request.">
        <Button size="sm" onClick={() => setLive((l) => !l)} aria-pressed={live}>
          {live ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          {live ? "Live" : "Paused"}
        </Button>
        <Button size="sm" onClick={download} disabled={!rows.length}>
          <Download className="size-3.5" /> CSV
        </Button>
      </PageHeader>

      {error && <p className="text-sm text-bad">Could not load requests: {error}</p>}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Distribution"
            subtitle={`${num(values.length)} successful requests in the current filter`}
            action={<Segmented size="sm" label="Metric" value={metric} onChange={setMetric} options={METRICS} />}
          />
          <div className="px-5 pt-6 pb-5">
            <Histogram
              values={values}
              color={m.color}
              format={m.fmt}
              markers={values.length > 1 ? [{ at: p50, label: "p50" }, { at: p95, label: "p95" }] : []}
            />
          </div>
        </Card>
        <Card>
          <CardHeader title="Filtered summary" />
          <div className="grid grid-cols-2 gap-2 px-5 pb-5">
            {[
              ["Requests", num(rows.length)],
              ["Errors", num(rows.filter((r) => r.status >= 400 && r.status !== 499).length)],
              [`${m.label} p50`, m.fmt(p50)],
              [`${m.label} p95`, m.fmt(p95)],
              ["Output tokens", num(rows.reduce((a, r) => a + r.completion_tokens, 0))],
              ["Prompt tokens", num(rows.reduce((a, r) => a + r.prompt_tokens, 0))],
            ].map(([k, v]) => (
              <div key={k} className="rounded-xl border border-line bg-subtle/40 p-3">
                <div className="text-[11px] text-muted">{k}</div>
                <div className="mt-0.5 text-base font-semibold tabular">{v}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <div className="flex flex-wrap items-center gap-3 px-5 py-4">
          <div className="relative min-w-56 flex-1">
            <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-faint" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by model, endpoint or status…"
              className="h-9 w-full rounded-lg border border-line bg-panel pr-3 pl-9 text-sm outline-none placeholder:text-faint focus:border-accent"
            />
          </div>
          <Segmented
            size="sm"
            label="Outcome"
            value={outcome}
            onChange={setOutcome}
            options={[
              { value: "all", label: "All" },
              { value: "ok", label: "Succeeded" },
              { value: "error", label: "Failed" },
            ]}
          />
          <Segmented
            size="sm"
            label="Surface"
            value={surface}
            onChange={setSurface}
            options={[
              { value: "all", label: "Both APIs" },
              { value: "ollama", label: "Ollama" },
              { value: "openai", label: "OpenAI" },
            ]}
          />
        </div>
        {!data ? (
          <div className="space-y-2 px-5 pb-5">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="h-9" />
            ))}
          </div>
        ) : rows.length ? (
          <RequestsTable rows={rows} now={now} />
        ) : (
          <Empty icon={<ListTree className="size-5" />} title={data.recent.length ? "No request matches the filter" : "No requests recorded yet"}>
            Every call to /api/chat, /api/generate, /v1/chat/completions and /v1/completions is recorded here.
          </Empty>
        )}
      </Card>
    </div>
  );
}
