"use client";

// A deliberately small Markdown subset rendered as React elements — never as
// HTML strings — so model output cannot inject markup into the page.

import { Check, Copy } from "lucide-react";
import { useState, type ReactNode } from "react";

export function useCopy() {
  const [copied, setCopied] = useState(false);
  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard blocked; nothing useful to do */
    }
  };
  return [copied, copy] as const;
}

export function CopyButton({ text, label = "Copy", className = "" }: { text: string; label?: string; className?: string }) {
  const [copied, copy] = useCopy();
  return (
    <button
      type="button"
      onClick={() => copy(text)}
      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-muted transition-colors hover:bg-subtle hover:text-fg ${className}`}
    >
      {copied ? <Check className="size-3.5 text-ok" /> : <Copy className="size-3.5" />}
      {copied ? "Copied" : label}
    </button>
  );
}

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  return (
    <div className="my-3 overflow-hidden rounded-xl border border-line bg-subtle/70">
      <div className="flex items-center justify-between border-b border-line px-3 py-1">
        <span className="font-mono text-[11px] text-faint">{lang || "text"}</span>
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto px-4 py-3 font-mono text-[13px] leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function inline(text: string): ReactNode[] {
  return text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g).flatMap((part, i) => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return [
        <code key={i} className="rounded bg-subtle px-1 py-0.5 font-mono text-[0.88em]">
          {part.slice(1, -1)}
        </code>,
      ];
    }
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      return [<strong key={i}>{part.slice(2, -2)}</strong>];
    }
    return part.split("\n").flatMap((line, j) => (j ? [<br key={`${i}-${j}`} />, line] : [line]));
  });
}

const BULLET = /^\s*[-*•]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;

function block(par: string, key: number): ReactNode {
  const lines = par.split("\n");
  const heading = /^(#{1,4})\s+(.*)$/.exec(par);
  if (heading && lines.length === 1) {
    return (
      <p key={key} className="mt-4 mb-1 font-semibold first:mt-0">
        {inline(heading[2])}
      </p>
    );
  }
  if (lines.every((l) => BULLET.test(l))) {
    return (
      <ul key={key} className="my-2 list-disc space-y-1 pl-5">
        {lines.map((l, i) => (
          <li key={i}>{inline(l.replace(BULLET, ""))}</li>
        ))}
      </ul>
    );
  }
  if (lines.every((l) => NUMBERED.test(l))) {
    return (
      <ol key={key} className="my-2 list-decimal space-y-1 pl-5">
        {lines.map((l, i) => (
          <li key={i}>{inline(l.replace(NUMBERED, ""))}</li>
        ))}
      </ol>
    );
  }
  return (
    <p key={key} className="my-2 first:mt-0 last:mb-0">
      {inline(par)}
    </p>
  );
}

export function Markdown({ text }: { text: string }) {
  return (
    <>
      {text.split("```").map((part, i) => {
        if (i % 2 === 1) {
          const nl = part.indexOf("\n");
          const lang = nl >= 0 ? part.slice(0, nl).trim() : "";
          const code = (nl >= 0 ? part.slice(nl + 1) : part).replace(/\n$/, "");
          return <CodeBlock key={i} code={code} lang={lang} />;
        }
        return part
          .split(/\n{2,}/)
          .filter((p) => p.trim())
          .map((p, j) => block(p.replace(/^\n+|\n+$/g, ""), i * 1000 + j));
      })}
    </>
  );
}
