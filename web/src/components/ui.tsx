import type { ComponentProps, ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Card({ className, ...props }: ComponentProps<"section">) {
  return (
    <section
      className={cx("rounded-2xl border border-line bg-panel shadow-card", className)}
      {...props}
    />
  );
}

export function CardHeader({
  title,
  subtitle,
  icon,
  action,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-3 px-5 pt-4 pb-3">
      <div className="flex min-w-0 items-center gap-2.5">
        {icon && <span className="text-muted">{icon}</span>}
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold tracking-tight">{title}</h2>
          {subtitle && <p className="truncate text-xs text-muted">{subtitle}</p>}
        </div>
      </div>
      {action}
    </header>
  );
}

const TONES = {
  neutral: "bg-subtle text-muted border-line",
  accent: "bg-accent-soft text-accent border-transparent",
  ok: "bg-[color-mix(in_srgb,var(--ok)_12%,transparent)] text-ok border-transparent",
  warn: "bg-[color-mix(in_srgb,var(--warn)_14%,transparent)] text-warn border-transparent",
  bad: "bg-[color-mix(in_srgb,var(--bad)_12%,transparent)] text-bad border-transparent",
  info: "bg-[color-mix(in_srgb,var(--info)_12%,transparent)] text-info border-transparent",
  violet: "bg-[color-mix(in_srgb,var(--violet)_12%,transparent)] text-violet border-transparent",
} as const;

export type Tone = keyof typeof TONES;

export function Badge({
  tone = "neutral",
  className,
  ...props
}: ComponentProps<"span"> & { tone?: Tone }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4 whitespace-nowrap",
        TONES[tone],
        className,
      )}
      {...props}
    />
  );
}

export function statusTone(status: number): Tone {
  if (status === 499) return "warn";
  if (status >= 500) return "bad";
  if (status >= 400) return "warn";
  return "ok";
}

const BUTTONS = {
  primary: "bg-accent text-accent-fg hover:bg-accent-hover shadow-sm",
  ghost: "border border-line bg-panel text-fg hover:bg-subtle",
  quiet: "text-muted hover:text-fg hover:bg-subtle",
  danger: "bg-bad text-white hover:opacity-90",
} as const;

export function Button({
  variant = "ghost",
  size = "md",
  className,
  ...props
}: ComponentProps<"button"> & { variant?: keyof typeof BUTTONS; size?: "sm" | "md" }) {
  return (
    <button
      type="button"
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:pointer-events-none disabled:opacity-45",
        size === "sm" ? "h-7 px-2.5 text-xs" : "h-9 px-3.5 text-sm",
        BUTTONS[variant],
        className,
      )}
      {...props}
    />
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  size = "md",
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode; hint?: string }[];
  onChange: (value: T) => void;
  size?: "sm" | "md";
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line bg-subtle p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          title={o.hint}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md font-medium transition-all",
            size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm",
            o.value === value ? "bg-panel text-fg shadow-sm" : "text-muted hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon && <div className="mb-1 rounded-xl bg-subtle p-3 text-muted">{icon}</div>}
      <p className="text-sm font-medium">{title}</p>
      {children && <div className="max-w-sm text-xs text-muted">{children}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("animate-pulse rounded-md bg-subtle", className)} />;
}

export function Kv({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 truncate text-right font-medium tabular">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function PageHeader({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code className="rounded bg-subtle px-1.5 py-0.5 font-mono text-[0.85em]">{children}</code>;
}
