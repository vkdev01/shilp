"use client";

import { Activity, Boxes, BookOpen, Gauge, ListTree, MessagesSquare, Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import type { Health } from "@/lib/api";
import { bytes } from "@/lib/format";
import { usePoll } from "@/lib/hooks";
import { cx } from "./ui";

const NAV = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/chat/", label: "Playground", icon: MessagesSquare },
  { href: "/models/", label: "Models", icon: Boxes },
  { href: "/requests/", label: "Requests", icon: ListTree },
];

const EXTERNAL = [
  { href: "/docs", label: "API docs", icon: BookOpen },
  { href: "/metrics", label: "Prometheus", icon: Activity },
];

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <path d="M9 22l7-12 7 12z" fill="var(--accent-fg)" />
    </svg>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);
  useEffect(() => setTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light"), []);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("shilp-theme", next);
    } catch {
      /* not persisted */
    }
    setTheme(next);
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle colour theme"
      title="Toggle colour theme"
      className="grid size-8 place-items-center rounded-lg text-muted transition-colors hover:bg-subtle hover:text-fg"
    >
      {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  );
}

function ServerStatus({ health, compact = false }: { health: ReturnType<typeof usePoll<Health>>; compact?: boolean }) {
  const { data, error, updatedAt } = health;
  const online = Boolean(data) && !(error && (!updatedAt || Date.now() - updatedAt > 10_000));
  const dot = (
    <span className={cx("live-dot size-2 rounded-full", online ? "bg-ok text-ok" : "bg-bad text-bad")} />
  );
  if (compact) return <span title={online ? "Server online" : "Server unreachable"}>{dot}</span>;

  return (
    <div className="rounded-xl border border-line bg-subtle/60 p-3 text-xs">
      <div className="flex items-center gap-2 font-medium">
        {dot}
        {online ? "Serving" : data ? "Unreachable" : "Connecting…"}
        {data && <span className="ml-auto font-mono text-faint">v{data.version}</span>}
      </div>
      {data && (
        <div className="mt-2 space-y-1 text-muted">
          <div className="flex justify-between">
            <span>Engine</span>
            <span className="font-mono text-fg">{data.engine}</span>
          </div>
          <div className="flex justify-between">
            <span>Store</span>
            <span className="tabular text-fg">
              {data.models} · {bytes(data.store_bytes)}
            </span>
          </div>
          <div className="flex justify-between gap-2">
            <span>Resident</span>
            <span className="truncate font-mono text-fg" title={data.resident.map((r) => r.ref).join(", ")}>
              {data.resident.length ? data.resident.map((r) => r.ref).join(", ") : "none"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function isActive(pathname: string, href: string) {
  const p = pathname.replace(/\/+$/, "") || "/";
  const h = href.replace(/\/+$/, "") || "/";
  return h === "/" ? p === "/" : p === h || p.startsWith(`${h}/`);
}

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "/";
  const health = usePoll<Health>("/health", 5000);

  return (
    <div className="flex min-h-dvh">
      <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col border-r border-line bg-panel/70 px-3 py-4 backdrop-blur lg:flex">
        <Link href="/" className="mb-6 flex items-center gap-2.5 px-2">
          <Logo />
          <span className="text-base font-semibold tracking-tight">shilp</span>
          <span className="rounded-md bg-subtle px-1.5 py-0.5 text-[10px] font-medium text-muted">serve</span>
        </Link>

        <nav aria-label="Main" className="space-y-0.5">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = isActive(pathname, href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors",
                  active ? "bg-accent-soft text-accent" : "text-muted hover:bg-subtle hover:text-fg",
                )}
              >
                <Icon className="size-4" />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-6 px-2.5 text-[11px] font-semibold uppercase tracking-wider text-faint">Raw</div>
        <nav aria-label="Raw endpoints" className="mt-1 space-y-0.5">
          {EXTERNAL.map(({ href, label, icon: Icon }) => (
            <a
              key={href}
              href={href}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-muted transition-colors hover:bg-subtle hover:text-fg"
            >
              <Icon className="size-4" />
              {label}
              <span className="ml-auto font-mono text-[10px] text-faint">{href}</span>
            </a>
          ))}
        </nav>

        <div className="mt-auto space-y-3">
          <ServerStatus health={health} />
          <div className="flex items-center justify-between px-1 text-xs text-faint">
            <span>Ollama + OpenAI APIs</span>
            <ThemeToggle />
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-line bg-panel/80 px-4 backdrop-blur lg:hidden">
          <Link href="/" className="flex items-center gap-2">
            <Logo />
            <span className="font-semibold">shilp</span>
          </Link>
          <nav aria-label="Main" className="flex flex-1 items-center gap-1 overflow-x-auto">
            {NAV.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                aria-label={label}
                className={cx(
                  "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm",
                  isActive(pathname, href) ? "bg-accent-soft text-accent" : "text-muted",
                )}
              >
                <Icon className="size-4" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            ))}
          </nav>
          <ServerStatus health={health} compact />
          <ThemeToggle />
        </header>
        <main className="min-h-0 flex-1">{children}</main>
      </div>
    </div>
  );
}
