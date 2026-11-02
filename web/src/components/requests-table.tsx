import type { RequestRow } from "@/lib/api";
import { ago, clock, ms, num, rate } from "@/lib/format";
import { Badge, statusTone } from "./ui";

export function RequestsTable({ rows, now, compact = false }: { rows: RequestRow[]; now: number; compact?: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-y border-line bg-subtle/50 text-left text-[11px] font-semibold uppercase tracking-wider text-faint">
            <th className="px-5 py-2 font-semibold">Time</th>
            <th className="px-3 py-2 font-semibold">Request</th>
            <th className="px-3 py-2 font-semibold">Model</th>
            <th className="px-3 py-2 font-semibold">Status</th>
            {!compact && <th className="px-3 py-2 text-right font-semibold">Prompt</th>}
            <th className="px-3 py-2 text-right font-semibold">Output</th>
            <th className="px-3 py-2 text-right font-semibold">TTFT</th>
            <th className="px-3 py-2 text-right font-semibold">Total</th>
            <th className="px-5 py-2 text-right font-semibold">tok/s</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line whitespace-nowrap">
          {rows.map((r) => (
            <tr key={r.id} className="transition-colors hover:bg-subtle/50">
              <td className="whitespace-nowrap px-5 py-2.5 tabular text-muted" title={new Date(r.ts * 1000).toLocaleString()}>
                {compact ? ago(r.ts, now) : clock(r.ts, true)}
              </td>
              <td className="whitespace-nowrap px-3 py-2.5">
                <div className="flex items-center gap-1.5">
                  <Badge tone={r.surface === "openai" ? "info" : "violet"}>{r.surface}</Badge>
                  <span className="font-mono text-xs">{r.endpoint}</span>
                  {r.stream && <span className="text-[10px] uppercase tracking-wide text-faint">stream</span>}
                </div>
              </td>
              <td className="max-w-48 truncate px-3 py-2.5 font-mono text-xs" title={r.model}>
                {r.model}
              </td>
              <td className="px-3 py-2.5">
                <Badge tone={statusTone(r.status)}>{r.status === 499 ? "499 closed" : r.status}</Badge>
              </td>
              {!compact && <td className="px-3 py-2.5 text-right tabular text-muted">{num(r.prompt_tokens)}</td>}
              <td className="px-3 py-2.5 text-right tabular">{num(r.completion_tokens)}</td>
              <td className="px-3 py-2.5 text-right tabular">{ms(r.ttft_ms)}</td>
              <td className="px-3 py-2.5 text-right tabular">{ms(r.total_ms)}</td>
              <td className="px-5 py-2.5 text-right font-medium tabular">{rate(r.tok_per_s)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
