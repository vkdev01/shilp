"use client";

import { Boxes, ChevronRight, Cpu, MessagesSquare, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

import { postJSON, type ShowResponse, type Stats, type TagModel } from "@/lib/api";
import { bytes, compact, duration, ms, rate } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import { CopyButton } from "./markdown";
import { Badge, Button, Card, CardHeader, Code, Empty, Kv, PageHeader, Segmented, Skeleton, cx } from "./ui";

function Details({ name, onClose }: { name: string; onClose: () => void }) {
  const [show, setShow] = useState<ShowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"info" | "shilpfile" | "template">("info");

  useEffect(() => {
    const abort = new AbortController();
    setShow(null);
    setError(null);
    postJSON<ShowResponse>("/api/show", { model: name }, { signal: abort.signal })
      .then(setShow)
      .catch((e) => !abort.signal.aborted && setError(e instanceof Error ? e.message : String(e)));
    return () => abort.abort();
  }, [name]);

  const text = tab === "shilpfile" ? show?.modelfile : show?.template || "(no template — the GGUF's own chat template is used)";

  return (
    <Card>
      <CardHeader
        icon={<Cpu className="size-4" />}
        title={<span className="font-mono">{name}</span>}
        subtitle="POST /api/show"
        action={
          <div className="flex items-center gap-2">
            <Segmented
              size="sm"
              label="Section"
              value={tab}
              onChange={setTab}
              options={[
                { value: "info", label: "Metadata" },
                { value: "shilpfile", label: "Shilpfile" },
                { value: "template", label: "Template" },
              ]}
            />
            <button type="button" onClick={onClose} className="rounded-md p-1 text-muted hover:bg-subtle" aria-label="Close details">
              <X className="size-4" />
            </button>
          </div>
        }
      />
      <div className="px-5 pb-5">
        {error ? (
          <p className="text-sm text-bad">{error}</p>
        ) : !show ? (
          <Skeleton className="h-40" />
        ) : tab === "info" ? (
          <div className="grid gap-6 md:grid-cols-2">
            <Kv
              items={Object.entries(show.model_info)
                .filter(([, v]) => v == null || typeof v !== "object")
                .map(([k, v]): [ReactNode, ReactNode] => [<span key={k} className="font-mono text-xs">{k}</span>, String(v ?? "—")])}
            />
            <div className="space-y-3">
              <div>
                <div className="mb-1.5 text-xs font-medium text-muted">Capabilities</div>
                <div className="flex flex-wrap gap-1">
                  {show.capabilities.map((c) => (
                    <Badge key={c} tone="accent">
                      {c}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-1.5 text-xs font-medium text-muted">Default parameters</div>
                <pre className="rounded-xl border border-line bg-subtle/60 p-3 font-mono text-xs">
                  {show.parameters || "(engine defaults)"}
                </pre>
              </div>
            </div>
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-line">
            <div className="flex justify-end border-b border-line bg-subtle/60 px-2 py-1">
              <CopyButton text={text ?? ""} />
            </div>
            <pre className="max-h-96 overflow-auto p-4 font-mono text-xs leading-relaxed">{text}</pre>
          </div>
        )}
      </div>
    </Card>
  );
}

export function Models() {
  const tags = usePoll<{ models: TagModel[] }>("/api/tags", 15_000);
  const stats = usePoll<Stats>("/api/stats?window=86400&recent=0", 5000);
  const [selected, setSelected] = useState<string | null>(null);

  const models = tags.data?.models;
  const resident = new Map((stats.data?.resident ?? []).map((r) => [r.ref, r]));
  const usage = new Map((stats.data?.window.by_model ?? []).map((g) => [g.model, g]));
  const total = models?.reduce((a, m) => a + m.size, 0) ?? 0;

  return (
    <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
      <PageHeader
        title="Models"
        subtitle={
          models
            ? `${models.length} stored · ${bytes(total)} on disk · ${resident.size} resident in memory`
            : "Reading the store…"
        }
      />

      {tags.error && <p className="text-sm text-bad">Could not list models: {tags.error}</p>}

      {!models ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }, (_, i) => (
            <Skeleton key={i} className="h-52 rounded-2xl" />
          ))}
        </div>
      ) : models.length === 0 ? (
        <Card>
          <Empty icon={<Boxes className="size-6" />} title="The store is empty">
            Pull a GGUF from the Hugging Face Hub: <Code>shilp pull hf:org/model-GGUF:q4_k_m</Code> or <Code>make pull</Code>.
          </Empty>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {models.map((m) => {
            const r = resident.get(m.name);
            const u = usage.get(m.name);
            return (
              <Card key={m.name} className={cx("flex flex-col p-5 transition-colors", selected === m.name && "border-accent")}>
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h2 className="truncate font-mono text-sm font-semibold" title={m.name}>
                      {m.name}
                    </h2>
                    <p className="mt-0.5 font-mono text-[11px] text-faint">{m.digest.slice(0, 19)}</p>
                  </div>
                  {r ? (
                    <Badge tone={r.refs ? "accent" : "ok"}>
                      <span className={cx("size-1.5 rounded-full", r.refs ? "live-dot bg-accent text-accent" : "bg-ok")} />
                      {r.refs ? `${r.refs} streaming` : "resident"}
                    </Badge>
                  ) : (
                    <Badge>on disk</Badge>
                  )}
                </div>

                <div className="mt-3 flex flex-wrap gap-1">
                  {[m.details.quantization_level, m.details.format, m.details.family, m.details.parameter_size]
                    .filter((v) => v && v !== "unknown")
                    .map((v, i) => (
                      <Badge key={v} tone={i === 0 ? "violet" : "neutral"}>
                        {v}
                      </Badge>
                    ))}
                </div>

                <dl className="mt-4 grid grid-cols-3 gap-3 border-t border-line pt-4 text-xs">
                  <div>
                    <dt className="text-faint">Size</dt>
                    <dd className="mt-0.5 font-semibold tabular">{bytes(m.size)}</dd>
                  </div>
                  <div>
                    <dt className="text-faint">Requests 24h</dt>
                    <dd className="mt-0.5 font-semibold tabular">{compact(u?.requests ?? 0)}</dd>
                  </div>
                  <div>
                    <dt className="text-faint">Decode</dt>
                    <dd className="mt-0.5 font-semibold tabular">{u?.tok_per_s ? `${rate(u.tok_per_s)} tok/s` : "—"}</dd>
                  </div>
                  <div>
                    <dt className="text-faint">TTFT p50</dt>
                    <dd className="mt-0.5 font-semibold tabular">{ms(u?.ttft_p50_ms)}</dd>
                  </div>
                  <div>
                    <dt className="text-faint">Loaded for</dt>
                    <dd className="mt-0.5 font-semibold tabular">{r ? duration(r.loaded_seconds) : "—"}</dd>
                  </div>
                  <div>
                    <dt className="text-faint">Tokens 24h</dt>
                    <dd className="mt-0.5 font-semibold tabular">{compact(u?.completion_tokens ?? 0)}</dd>
                  </div>
                </dl>

                <div className="mt-auto flex gap-2 pt-5">
                  <Link
                    href={`/chat/?model=${encodeURIComponent(m.name)}`}
                    className="inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-lg bg-accent px-3 text-sm font-medium text-accent-fg hover:bg-accent-hover"
                  >
                    <MessagesSquare className="size-3.5" /> Chat
                  </Link>
                  <Button size="sm" className="h-8 flex-1" onClick={() => setSelected(selected === m.name ? null : m.name)}>
                    Details <ChevronRight className={cx("size-3.5 transition-transform", selected === m.name && "rotate-90")} />
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {selected && <Details name={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
