// Shapes of the JSON the server returns. Kept by hand: the API is small, and a
// type here that disagrees with the server is caught the first time a page renders.

export type Dist = { avg: number; p50: number; p90: number; p95: number; p99: number; max: number };

export type Bucket = {
  ts: number;
  requests: number;
  errors: number;
  tokens: number;
  ttft_p50_ms: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  tok_per_s: number;
};

export type Group = {
  model?: string;
  surface?: string;
  endpoint?: string;
  requests: number;
  errors: number;
  completion_tokens: number;
  ttft_p50_ms: number;
  latency_p95_ms: number;
  tok_per_s: number;
};

export type RequestRow = {
  id: number;
  ts: number;
  surface: string;
  endpoint: string;
  model: string;
  stream: boolean;
  status: number;
  prompt_tokens: number;
  completion_tokens: number;
  ttft_ms: number;
  total_ms: number;
  tok_per_s: number;
};

export type Resident = {
  ref: string;
  size: number;
  refs: number;
  idle_seconds: number;
  loaded_seconds: number;
};

export type Window = {
  seconds: number;
  bucket_seconds: number;
  requests: number;
  errors: number;
  cancelled: number;
  error_rate: number;
  rps: number;
  prompt_tokens: number;
  completion_tokens: number;
  output_tok_per_s: number;
  ttft_ms: Dist;
  latency_ms: Dist;
  tok_per_s: Dist;
  by_model: Group[];
  by_surface: Group[];
  by_endpoint: Group[];
  by_status: Record<string, number>;
  series: Bucket[];
};

export type Stats = {
  server: {
    version: string;
    engine: string;
    host: string;
    port: number;
    max_resident: number;
    gpu_layers: number;
    threads: number | null;
    default_model: string | null;
    python: string;
    platform: string;
    started_at: number;
    uptime_s: number;
  };
  store: { models: number; blobs: number; bytes: number };
  resident: Resident[];
  in_flight: number;
  lifetime: {
    requests: number;
    prompt_tokens: number;
    completion_tokens: number;
    avg_ttft_ms: number;
    avg_tok_per_s: number;
    best_tok_per_s: number;
  };
  window: Window;
  recent: RequestRow[];
};

export type Health = {
  status: string;
  version: string;
  engine: string;
  models: number;
  store_bytes: number;
  resident: Resident[];
};

export type TagModel = {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    parameter_size: string;
    quantization_level: string;
  };
};

export type ShowResponse = {
  modelfile: string;
  parameters: string;
  template: string;
  details: TagModel["details"];
  model_info: Record<string, unknown>;
  capabilities: string[];
};

export async function errorMessage(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const obj = JSON.parse(text);
    if (obj.error) return typeof obj.error === "string" ? obj.error : obj.error.message;
    if (obj.detail) return typeof obj.detail === "string" ? obj.detail : JSON.stringify(obj.detail);
  } catch {
    /* not JSON */
  }
  return text || `HTTP ${res.status}`;
}

export async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { accept: "application/json", ...init?.headers } });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as T;
}

export async function postJSON<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
  return getJSON<T>(path, {
    ...init,
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}
