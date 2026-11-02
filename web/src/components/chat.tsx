"use client";

import {
  ArrowUp,
  Bot,
  Info,
  PanelLeft,
  RefreshCw,
  RotateCcw,
  Square,
  Terminal,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { errorMessage, type TagModel } from "@/lib/api";
import {
  buildRequest,
  parseEvent,
  parseFull,
  readLines,
  toCurl,
  type ChatSettings,
  type ChatTurn,
  type Surface,
  type Usage,
} from "@/lib/chat";
import { bytes, ms, num, rate } from "@/lib/format";
import { usePoll, useStored } from "@/lib/hooks";
import { Sparkline } from "./charts";
import { CopyButton, Markdown } from "./markdown";
import { Badge, Button, Segmented, cx } from "./ui";

const RAW_LIMIT = 20_000;

const DEFAULTS: ChatSettings = {
  surface: "ollama",
  model: "",
  system: "",
  temperature: 0.7,
  topP: 0.95,
  maxTokens: null,
  stream: true,
};

const SUGGESTIONS = [
  "Explain what a KV cache is in two sentences.",
  "Why is time to first token so different from decode speed?",
  "Write a haiku about quantization.",
  "Give me a Python one-liner to reverse a string.",
];

type Status = "pending" | "streaming" | "done" | "error" | "stopped";
type Timing = { start: number; first?: number; end?: number; arrivals: number[] };
type Msg = {
  id: number;
  role: "user" | "assistant";
  content: string;
  status: Status;
  /** A user turn whose reply failed: shown, but not resent as history. */
  excluded?: boolean;
  model?: string;
  surface?: Surface;
  stream?: boolean;
  timing?: Timing;
  usage?: Usage;
};

type Derived = {
  tokens: number;
  prompt?: number;
  ttft?: number;
  total: number;
  tps?: number;
  reason?: string;
};

function derive(m: Msg, now: number): Derived | null {
  const t = m.timing;
  if (!t) return null;
  const end = t.end ?? now;
  const tokens = m.usage?.completion ?? t.arrivals.length;
  let tps: number | undefined;
  if (tokens && t.first != null && end > t.first) tps = tokens / ((end - t.first) / 1000);
  else if (tokens && m.usage?.evalNs) tps = tokens / (m.usage.evalNs / 1e9);
  return {
    tokens,
    prompt: m.usage?.prompt,
    ttft: t.first != null ? t.first - t.start : undefined,
    total: end - t.start,
    tps,
    reason: m.usage?.reason,
  };
}

/** Tokens per slice of the generation, to show how steady decoding was. */
function cadence(t: Timing | undefined, slices = 36): number[] {
  if (!t || t.first == null || t.end == null || t.arrivals.length < 4) return [];
  const span = Math.max(1, t.end - t.first);
  const out = new Array(slices).fill(0) as number[];
  for (const a of t.arrivals) out[Math.min(slices - 1, Math.floor(((a - t.first) / span) * slices))] += 1;
  return out;
}

function toTurns(messages: Msg[]): ChatTurn[] {
  return messages
    .filter((m) => !m.excluded && (m.role === "user" || ((m.status === "done" || m.status === "stopped") && m.content)))
    .map((m) => ({ role: m.role, content: m.content }));
}

/** `performance.now()`, re-read every 200 ms while `active`, for live timers. */
function usePerfClock(active: boolean) {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!active) return;
    setNow(performance.now());
    const id = setInterval(() => setNow(performance.now()), 200);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

function Field({ label, children, aside }: { label: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted">{label}</span>
        {aside}
      </div>
      {children}
    </div>
  );
}

const input =
  "w-full rounded-lg border border-line bg-panel px-3 py-2 text-sm outline-none transition-shadow placeholder:text-faint focus:border-accent focus:ring-3 focus:ring-accent/20 disabled:opacity-50";

function Slider({ value, min, max, step, onChange, label }: { value: number; min: number; max: number; step: number; onChange: (v: number) => void; label: string }) {
  return (
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full accent-[var(--accent)]"
    />
  );
}

function TimingBar({ d }: { d: Derived }) {
  if (d.ttft == null || d.total <= 0) return null;
  const f = Math.min(1, d.ttft / d.total);
  return (
    <div className="flex items-center gap-2 text-[10px] text-faint">
      <div className="flex h-1.5 w-40 overflow-hidden rounded-full bg-subtle" title="prefill (time to first token) vs decode">
        <div style={{ width: `${Math.max(2, f * 100)}%`, background: "var(--violet)" }} />
        <div className="flex-1" style={{ background: "var(--teal)" }} />
      </div>
      <span>
        <span className="text-violet">prefill {Math.round(f * 100)}%</span> ·{" "}
        <span className="text-teal">decode {Math.round((1 - f) * 100)}%</span>
      </span>
    </div>
  );
}

function AssistantMeta({ m, now }: { m: Msg; now: number }) {
  const d = derive(m, now);
  if (!d) return null;
  const live = m.status === "streaming" || m.status === "pending";
  const spark = cadence(m.timing);
  return (
    <div className="mt-3 space-y-2">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <Badge tone={m.surface === "openai" ? "info" : "violet"}>{m.surface === "openai" ? "OpenAI" : "Ollama"}</Badge>
        {live && d.ttft == null && <Badge>waiting {ms(d.total)}</Badge>}
        {d.tokens > 0 && (
          <Badge>
            {num(d.tokens)} tok{d.prompt != null && <span className="text-faint"> · {num(d.prompt)} in</span>}
          </Badge>
        )}
        {d.tps != null && <Badge tone="accent">{rate(d.tps)} tok/s</Badge>}
        {d.ttft != null && <Badge>TTFT {ms(d.ttft)}</Badge>}
        {!live && <Badge>{ms(d.total)}</Badge>}
        {m.status === "stopped" && <Badge tone="warn">stopped</Badge>}
        {d.reason && d.reason !== "stop" && <Badge tone="warn">{d.reason}</Badge>}
        {!m.stream && <Badge>buffered</Badge>}
        {!live && m.content && <CopyButton text={m.content} className="ml-auto" />}
      </div>
      {!live && (
        <div className="flex flex-wrap items-center gap-4">
          <TimingBar d={d} />
          {spark.length > 0 && (
            <div className="flex items-center gap-2 text-[10px] text-faint" title="Tokens arriving over the generation">
              <div className="h-4 w-28">
                <Sparkline kind="bar" values={spark} color="var(--teal)" height={16} />
              </div>
              cadence
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Message({ m, now }: { m: Msg; now: number }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div
          className={cx(
            "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-inset px-4 py-2.5 text-[15px] leading-relaxed",
            m.excluded && "opacity-60",
          )}
        >
          {m.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
        <Bot className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 font-mono text-[11px] text-faint">{m.model}</div>
        {m.status === "pending" ? (
          <div className="flex h-6 items-center gap-1" aria-label="Waiting for the first token">
            {[0, 1, 2].map((i) => (
              <span key={i} className="size-1.5 animate-bounce rounded-full bg-faint" style={{ animationDelay: `${i * 120}ms` }} />
            ))}
          </div>
        ) : m.status === "error" ? (
          <div className="rounded-xl border border-bad/30 bg-bad/10 px-3 py-2 text-sm text-bad">{m.content}</div>
        ) : (
          <div className={cx("text-[15px] leading-relaxed", m.status === "streaming" && "caret")}>
            {m.content ? <Markdown text={m.content} /> : <span className="text-muted">(empty response)</span>}
          </div>
        )}
        {m.status !== "error" && <AssistantMeta m={m} now={now} />}
      </div>
    </div>
  );
}

export function Chat() {
  const [settings, setSettings, ready] = useStored<ChatSettings>("shilp-chat-settings", DEFAULTS);
  const tags = usePoll<{ models: TagModel[] }>("/api/tags", 30_000);
  const models = tags.data?.models;

  const [messages, setMessages] = useState<Msg[]>([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [tab, setTab] = useState<"metrics" | "request" | "response">("metrics");
  const [inspector, setInspector] = useState({ url: "", status: "", curl: "", raw: "" });

  const now = usePerfClock(busy);
  const controller = useRef<AbortController | null>(null);
  const conversation = useRef(0);
  const ids = useRef(0);
  const scroller = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const urlModel = useRef<string | null>(null);

  const set = <K extends keyof ChatSettings>(key: K, value: ChatSettings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  useEffect(() => {
    urlModel.current = new URLSearchParams(location.search).get("model");
  }, []);

  // Pick a model once both the list and the saved settings are known.
  useEffect(() => {
    if (!ready || !models) return;
    const names = models.map((m) => m.name);
    const wanted = urlModel.current;
    urlModel.current = null;
    if (wanted && names.includes(wanted)) set("model", wanted);
    else if (!names.includes(settings.model)) set("model", names[0] ?? "");
  }, [ready, models]); // settings.model is read, not tracked: re-picking on every change would fight the user

  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [prompt]);

  const current = models?.find((m) => m.name === settings.model);
  const preview = useMemo(() => buildRequest(settings, []), [settings]);

  const session = useMemo(() => {
    const done = messages
      .filter((m) => m.role === "assistant" && (m.status === "done" || m.status === "stopped"))
      .map((m) => derive(m, 0))
      .filter((d): d is Derived => d != null);
    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined);
    const tps = done.map((d) => d.tps ?? 0);
    const ttft = done.map((d) => d.ttft ?? 0);
    const last = done[done.length - 1];
    return {
      turns: done.length,
      output: done.reduce((a, d) => a + d.tokens, 0),
      context: last ? (last.prompt ?? 0) + last.tokens : 0,
      avgTps: avg(tps.filter(Boolean)),
      bestTps: Math.max(0, ...tps),
      avgTtft: avg(ttft.filter(Boolean)),
      tps,
      ttft,
    };
  }, [messages]);

  async function send(text: string) {
    if (!settings.model || busy) return;
    const conv = conversation.current;
    const userId = ++ids.current;
    const botId = ++ids.current;
    const { url, body } = buildRequest(settings, [...toTurns(messages), { role: "user", content: text }]);
    const surface = settings.surface;
    const streaming = settings.stream;

    setInspector({ url, status: "sending…", curl: toCurl(url, body), raw: "" });
    const timing: Timing = { start: performance.now(), arrivals: [] };
    setMessages((ms) => [
      ...ms,
      { id: userId, role: "user", content: text, status: "done" },
      { id: botId, role: "assistant", content: "", status: "pending", model: settings.model, surface, stream: streaming, timing },
    ]);

    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);

    let answer = "";
    let raw = "";
    let usage: Usage | undefined;
    let frame = 0;
    let status: Status = "done";

    const patch = (changes: Partial<Msg>) => {
      if (conv !== conversation.current) return;
      setMessages((ms) => ms.map((m) => (m.id === botId ? { ...m, ...changes } : m)));
    };
    const paint = () => {
      frame = 0;
      patch({ content: answer, status: "streaming", timing: { ...timing } });
    };

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: abort.signal,
      });
      setInspector((i) => ({ ...i, status: `${res.status} ${res.statusText}`.trim() }));
      if (!res.ok) throw new Error(await errorMessage(res));

      if (streaming) {
        for await (const line of readLines(res)) {
          if (raw.length < RAW_LIMIT) raw += `${line}\n`;
          const event = parseEvent(surface, line);
          if (!event) continue;
          if (event.delta) {
            const t = performance.now();
            timing.first ??= t;
            timing.arrivals.push(t);
            answer += event.delta;
            if (!frame) frame = requestAnimationFrame(paint);
          }
          if (event.usage) usage = event.usage;
          if (event.done) break;
        }
      } else {
        const obj = await res.json();
        raw = JSON.stringify(obj, null, 2);
        const parsed = parseFull(surface, obj);
        answer = parsed.text;
        usage = parsed.usage;
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        status = "stopped";
      } else {
        if (frame) cancelAnimationFrame(frame);
        timing.end = performance.now();
        patch({ status: "error", content: err instanceof Error ? err.message : String(err), timing: { ...timing } });
        if (conv === conversation.current) {
          setMessages((ms) => ms.map((m) => (m.id === userId ? { ...m, excluded: true } : m)));
          setPrompt((p) => p || text);
        }
        setInspector((i) => ({ ...i, raw: raw || (err instanceof Error ? err.message : String(err)) }));
        finish();
        return;
      }
    }

    if (frame) cancelAnimationFrame(frame);
    timing.end = performance.now();
    patch({ content: answer, status, usage, timing: { ...timing } });
    setInspector((i) => ({ ...i, raw: raw.length >= RAW_LIMIT ? `${raw}\n… (truncated)` : raw }));
    finish();
  }

  function finish() {
    controller.current = null;
    setBusy(false);
    requestAnimationFrame(() => textarea.current?.focus());
  }

  function submit(text = prompt) {
    const t = text.trim();
    if (!t || busy || !settings.model) return;
    setPrompt("");
    void send(t);
  }

  function newChat() {
    conversation.current += 1;
    controller.current?.abort();
    setMessages([]);
    setPrompt("");
    setInspector({ url: "", status: "", curl: "", raw: "" });
    textarea.current?.focus();
  }

  const settingsPanel = (
    <div className="space-y-5">
      <Field label="API surface">
        <Segmented
          label="API surface"
          value={settings.surface}
          onChange={(v) => set("surface", v)}
          options={[
            { value: "ollama", label: "Ollama", hint: "POST /api/chat" },
            { value: "openai", label: "OpenAI", hint: "POST /v1/chat/completions" },
          ]}
        />
      </Field>

      <Field
        label="Model"
        aside={
          <button type="button" onClick={tags.refresh} className="flex items-center gap-1 text-xs text-muted hover:text-fg" title="Reload /api/tags">
            <RefreshCw className="size-3" /> Refresh
          </button>
        }
      >
        <select
          className={input}
          value={settings.model}
          disabled={!models?.length}
          onChange={(e) => set("model", e.target.value)}
          aria-label="Model"
        >
          {!models?.length && <option value="">{models ? "No models stored" : "Loading…"}</option>}
          {models?.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
        {models && !models.length && (
          <p className="text-xs text-muted">
            Pull one first: <code className="font-mono">make pull</code>
          </p>
        )}
        {current && (
          <div className="flex flex-wrap gap-1">
            {[current.details.quantization_level, current.details.family, current.details.parameter_size]
              .filter((v) => v && v !== "unknown")
              .map((v) => (
                <Badge key={v}>{v}</Badge>
              ))}
            <Badge>{bytes(current.size)}</Badge>
          </div>
        )}
      </Field>

      <Field label="System prompt">
        <textarea
          className={cx(input, "min-h-20 resize-y")}
          rows={3}
          value={settings.system}
          onChange={(e) => set("system", e.target.value)}
          placeholder="Optional — e.g. You are a terse technical assistant."
        />
      </Field>

      <Field label="Temperature" aside={<span className="font-mono text-xs tabular">{settings.temperature.toFixed(2)}</span>}>
        <Slider label="Temperature" value={settings.temperature} min={0} max={2} step={0.05} onChange={(v) => set("temperature", v)} />
      </Field>

      <Field label="Top p" aside={<span className="font-mono text-xs tabular">{settings.topP.toFixed(2)}</span>}>
        <Slider label="Top p" value={settings.topP} min={0.05} max={1} step={0.05} onChange={(v) => set("topP", v)} />
      </Field>

      <Field label="Max tokens">
        <input
          type="number"
          className={input}
          min={1}
          max={8192}
          placeholder="model default"
          value={settings.maxTokens ?? ""}
          onChange={(e) => {
            const n = parseInt(e.target.value, 10);
            set("maxTokens", Number.isFinite(n) && n > 0 ? n : null);
          }}
        />
      </Field>

      <label className="flex cursor-pointer items-center justify-between gap-3 text-sm">
        <span>
          Stream tokens
          <span className="block text-xs text-muted">{settings.surface === "ollama" ? "NDJSON" : "Server-sent events"}</span>
        </span>
        <input
          type="checkbox"
          checked={settings.stream}
          onChange={(e) => set("stream", e.target.checked)}
          className="peer sr-only"
        />
        <span className="relative h-5 w-9 rounded-full bg-inset transition-colors after:absolute after:top-0.5 after:left-0.5 after:size-4 after:rounded-full after:bg-panel after:shadow after:transition-transform peer-checked:bg-accent peer-checked:after:translate-x-4 peer-focus-visible:ring-2 peer-focus-visible:ring-accent" />
      </label>

      <Button className="w-full" onClick={newChat}>
        <RotateCcw className="size-4" /> New chat
      </Button>
    </div>
  );

  const inspectorPanel = (
    <div className="flex h-full flex-col">
      <div className="border-b border-line px-4 pt-4 pb-3">
        <Segmented
          size="sm"
          label="Inspector"
          value={tab}
          onChange={setTab}
          options={[
            { value: "metrics", label: "Metrics" },
            { value: "request", label: "Request" },
            { value: "response", label: "Response" },
          ]}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {tab === "metrics" && (
          <div className="space-y-5">
            <div>
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-faint">This conversation</h3>
              <div className="grid grid-cols-2 gap-2">
                {[
                  ["Turns", num(session.turns)],
                  ["Output tokens", num(session.output)],
                  ["Context used", session.context ? `${num(session.context)} tok` : "—"],
                  ["Avg TTFT", ms(session.avgTtft)],
                  ["Avg decode", session.avgTps ? `${rate(session.avgTps)} tok/s` : "—"],
                  ["Best decode", session.bestTps ? `${rate(session.bestTps)} tok/s` : "—"],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-xl border border-line bg-subtle/40 p-3">
                    <div className="text-[11px] text-muted">{k}</div>
                    <div className="mt-0.5 text-base font-semibold tabular">{v}</div>
                  </div>
                ))}
              </div>
            </div>
            {session.turns > 1 && (
              <div className="space-y-3">
                <div>
                  <div className="mb-1 flex justify-between text-xs text-muted">
                    <span>Decode speed per turn</span>
                    <span className="text-teal">tok/s</span>
                  </div>
                  <div className="h-10">
                    <Sparkline values={session.tps} color="var(--teal)" gaps height={40} />
                  </div>
                </div>
                <div>
                  <div className="mb-1 flex justify-between text-xs text-muted">
                    <span>TTFT per turn</span>
                    <span className="text-violet">ms</span>
                  </div>
                  <div className="h-10">
                    <Sparkline values={session.ttft} color="var(--violet)" gaps height={40} />
                  </div>
                </div>
              </div>
            )}
            <div className="rounded-xl border border-line bg-subtle/40 p-3 text-xs leading-relaxed text-muted">
              <Info className="mb-1 size-3.5" />
              <b className="text-fg">TTFT</b> is measured in the browser from send to the first token — it includes model
              load and prompt processing. <b className="text-fg">Decode</b> is output tokens over the time after the first
              token. Server-side numbers for every request are on the Overview page.
            </div>
          </div>
        )}
        {tab === "request" && (
          <div className="space-y-3">
            <p className="font-mono text-xs text-muted">
              {inspector.url ? `POST ${inspector.url} · ${inspector.status}` : `Next: POST ${preview.url}`}
            </p>
            <div className="overflow-hidden rounded-xl border border-line">
              <div className="flex items-center justify-between border-b border-line bg-subtle/60 px-3 py-1.5">
                <span className="flex items-center gap-1.5 text-xs text-muted">
                  <Terminal className="size-3.5" /> curl
                </span>
                {inspector.curl && <CopyButton text={inspector.curl} />}
              </div>
              <pre className="max-h-[60vh] overflow-auto p-3 font-mono text-[11px] leading-relaxed">
                {inspector.curl || "Send a message to see the exact request."}
              </pre>
            </div>
          </div>
        )}
        {tab === "response" && (
          <div className="overflow-hidden rounded-xl border border-line">
            <div className="flex items-center justify-between border-b border-line bg-subtle/60 px-3 py-1.5">
              <span className="text-xs text-muted">Raw {settings.stream ? (settings.surface === "ollama" ? "NDJSON" : "SSE") : "JSON"}</span>
              {inspector.raw && <CopyButton text={inspector.raw} />}
            </div>
            <pre className="max-h-[70vh] overflow-auto p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap break-all">
              {inspector.raw || "Nothing received yet."}
            </pre>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="grid h-[calc(100dvh-3.5rem)] lg:h-dvh lg:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_340px]">
      {(drawer || inspectorOpen) && (
        <div
          className="fixed inset-0 z-30 bg-black/30 xl:hidden"
          onClick={() => {
            setDrawer(false);
            setInspectorOpen(false);
          }}
        />
      )}
      <aside
        className={cx(
          "fixed inset-y-0 left-0 z-40 w-80 overflow-y-auto border-r border-line bg-panel p-5 transition-transform lg:static lg:z-auto lg:w-auto lg:translate-x-0",
          drawer ? "translate-x-0" : "-translate-x-full",
        )}
        aria-label="Chat settings"
      >
        <div className="mb-5 flex items-center justify-between">
          <h1 className="text-lg font-semibold tracking-tight">Playground</h1>
          <button type="button" className="text-muted lg:hidden" onClick={() => setDrawer(false)} aria-label="Close settings">
            <X className="size-5" />
          </button>
        </div>
        {settingsPanel}
      </aside>

      <section className="flex min-h-0 min-w-0 flex-col" aria-label="Conversation">
        <div className="flex items-center gap-2 border-b border-line px-4 py-2 xl:hidden">
          <Button size="sm" className="lg:hidden" onClick={() => setDrawer(true)}>
            <PanelLeft className="size-3.5" /> Settings
          </Button>
          <span className="truncate font-mono text-xs text-muted">{settings.model || "no model"}</span>
          <Button size="sm" className="ml-auto" onClick={() => setInspectorOpen(true)}>
            <Terminal className="size-3.5" /> Inspector
          </Button>
        </div>

        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto" aria-live="polite">
          {messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center px-6 text-center">
              <div className="mb-4 grid size-14 place-items-center rounded-2xl bg-accent-soft text-accent">
                <Bot className="size-7" />
              </div>
              <h2 className="text-xl font-semibold tracking-tight">Talk to your model</h2>
              <p className="mt-2 text-sm text-muted">
                Messages go to the API you pick — the same endpoints any client uses. Every reply shows its time to
                first token, decode speed and token counts; the inspector has the exact curl.
              </p>
              <div className="mt-6 grid w-full gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    disabled={!settings.model || busy}
                    onClick={() => submit(s)}
                    className="rounded-xl border border-line bg-panel px-4 py-3 text-left text-sm text-muted transition-colors hover:border-line-strong hover:text-fg disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 px-4 py-6 sm:px-6">
              {messages.map((m) => (
                <Message key={m.id} m={m} now={now} />
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-line bg-bg/80 px-4 py-3 backdrop-blur sm:px-6">
          <form
            className="mx-auto max-w-3xl"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <div className="flex items-end gap-2 rounded-2xl border border-line bg-panel p-2 shadow-card transition-shadow focus-within:border-accent focus-within:ring-3 focus-within:ring-accent/15">
              <textarea
                ref={textarea}
                rows={1}
                value={prompt}
                disabled={busy}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    submit();
                  }
                }}
                placeholder={settings.model ? "Send a message…" : "Pull a model to start chatting"}
                aria-label="Message"
                className="max-h-56 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] outline-none placeholder:text-faint disabled:opacity-60"
              />
              {busy ? (
                <Button variant="danger" onClick={() => controller.current?.abort()} aria-label="Stop generating">
                  <Square className="size-3.5 fill-current" /> Stop
                </Button>
              ) : (
                <Button type="submit" variant="primary" disabled={!prompt.trim() || !settings.model} aria-label="Send">
                  <ArrowUp className="size-4" />
                </Button>
              )}
            </div>
            <div className="mt-1.5 flex justify-between px-1 text-[11px] text-faint">
              <span className="font-mono">
                POST {preview.url} · {settings.stream ? "stream" : "buffered"}
              </span>
              <span>Enter to send · Shift+Enter for a new line</span>
            </div>
          </form>
        </div>
      </section>

      <aside
        className={cx(
          "fixed inset-y-0 right-0 z-40 w-[340px] max-w-full border-l border-line bg-panel transition-transform xl:static xl:z-auto xl:w-auto xl:translate-x-0 xl:bg-panel/60",
          inspectorOpen ? "translate-x-0" : "translate-x-full",
        )}
        aria-label="Inspector"
      >
        <button
          type="button"
          className="absolute top-4 right-4 text-muted xl:hidden"
          onClick={() => setInspectorOpen(false)}
          aria-label="Close inspector"
        >
          <X className="size-5" />
        </button>
        {inspectorPanel}
      </aside>
    </div>
  );
}
