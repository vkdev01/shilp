// Talking to both chat surfaces exactly as an external client would: build the
// request, read NDJSON (Ollama) or SSE (OpenAI), normalise both into one event.

export type Surface = "ollama" | "openai";

export type ChatTurn = { role: "system" | "user" | "assistant"; content: string };

export type ChatSettings = {
  surface: Surface;
  model: string;
  system: string;
  temperature: number;
  topP: number;
  maxTokens: number | null;
  stream: boolean;
};

export type Usage = {
  completion?: number;
  prompt?: number;
  evalNs?: number;
  reason?: string;
};

export type StreamEvent = { delta: string; done: boolean; usage: Usage | null };

export function buildRequest(settings: ChatSettings, history: ChatTurn[]) {
  const messages: ChatTurn[] = [];
  if (settings.system.trim()) messages.push({ role: "system", content: settings.system.trim() });
  messages.push(...history);

  if (settings.surface === "ollama") {
    const options: Record<string, number> = { temperature: settings.temperature, top_p: settings.topP };
    if (settings.maxTokens) options.num_predict = settings.maxTokens;
    return {
      url: "/api/chat",
      body: { model: settings.model, messages, stream: settings.stream, options },
    };
  }
  const body: Record<string, unknown> = {
    model: settings.model,
    messages,
    stream: settings.stream,
    temperature: settings.temperature,
    top_p: settings.topP,
  };
  if (settings.maxTokens) body.max_tokens = settings.maxTokens;
  return { url: "/v1/chat/completions", body };
}

export function toCurl(url: string, body: Record<string, unknown>): string {
  const json = JSON.stringify(body, null, 2).replace(/'/g, "'\\''");
  const flag = body.stream ? "-N " : "";
  return `curl ${flag}${location.origin}${url} \\\n  -H 'content-type: application/json' \\\n  -d '${json}'`;
}

export async function* readLines(response: Response): AsyncGenerator<string> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, index).replace(/\r$/, "");
      buffer = buffer.slice(index + 1);
      yield line;
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) yield buffer;
}

function errorText(error: unknown): string {
  if (typeof error === "string") return error;
  if (error && typeof error === "object" && "message" in error) return String(error.message);
  return JSON.stringify(error);
}

export function parseEvent(surface: Surface, line: string): StreamEvent | null {
  if (surface === "ollama") {
    if (!line.trim()) return null;
    const obj = JSON.parse(line);
    if (obj.error) throw new Error(errorText(obj.error));
    return {
      delta: obj.message?.content ?? "",
      done: Boolean(obj.done),
      usage: obj.done
        ? {
            completion: obj.eval_count,
            prompt: obj.prompt_eval_count,
            evalNs: obj.eval_duration,
            reason: obj.done_reason,
          }
        : null,
    };
  }
  if (!line.startsWith("data:")) return null;
  const payload = line.slice(5).trim();
  if (payload === "[DONE]") return { delta: "", done: true, usage: null };
  const obj = JSON.parse(payload);
  if (obj.error) throw new Error(errorText(obj.error));
  const choice = obj.choices?.[0] ?? {};
  return {
    delta: choice.delta?.content ?? "",
    done: false,
    usage: obj.usage
      ? {
          completion: obj.usage.completion_tokens,
          prompt: obj.usage.prompt_tokens,
          reason: choice.finish_reason,
        }
      : null,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- two wire shapes, narrowed by surface
export function parseFull(surface: Surface, obj: Record<string, any>): { text: string; usage: Usage } {
  if (surface === "ollama") {
    return {
      text: obj.message?.content ?? "",
      usage: {
        completion: obj.eval_count,
        prompt: obj.prompt_eval_count,
        evalNs: obj.eval_duration,
        reason: obj.done_reason,
      },
    };
  }
  const choice = obj.choices?.[0] ?? {};
  return {
    text: choice.message?.content ?? "",
    usage: {
      completion: obj.usage?.completion_tokens,
      prompt: obj.usage?.prompt_tokens,
      reason: choice.finish_reason,
    },
  };
}
