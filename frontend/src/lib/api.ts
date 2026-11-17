import type { ChatAnswer, Explain, Health, Report, Study, StudySummary } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export const assetUrl = (path: string | null | undefined) =>
  !path ? "" : path.startsWith("http") || path.startsWith("data:") ? path : `${API_URL}${path}`;

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let code = "http_error";
    let message = res.statusText;
    try {
      const j = await res.json();
      code = j?.error?.code ?? code;
      message = j?.error?.message ?? message;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/api/system/health"),
  reload: () => request<Record<string, unknown>>("/api/system/reload", { method: "POST" }),
  dashboard: () => request<Dashboard>("/api/dashboard"),
  studies: () => request<StudySummary[]>("/api/studies"),
  study: (id: string) => request<Study>(`/api/studies/${id}`),
  deleteStudy: (id: string) => request<{ deleted: string }>(`/api/studies/${id}`, { method: "DELETE" }),
  analyze: (id: string, force = false) =>
    request<Study>(`/api/studies/${id}/analyze?force=${force}`, { method: "POST" }),
  explain: (id: string, finding: string, method = "gradcam", cmap = "inferno") =>
    request<Explain>(
      `/api/studies/${id}/explain?finding=${encodeURIComponent(finding)}&method=${method}&cmap=${cmap}`,
    ),
  worklist: (params: { level?: string; q?: string; modality?: string } = {}) => {
    const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v) as [string, string][]);
    return request<{ items: StudySummary[]; counts: Record<string, number> }>(`/api/triage/worklist?${qs}`);
  },
  reportByStudy: (studyId: string) => request<Report>(`/api/reports/by-study/${studyId}`),
  generateReport: (studyId: string, engine = "rules", style = "standard") =>
    request<Report>(`/api/reports/${studyId}/generate?engine=${engine}&style=${style}`, { method: "POST" }),
  updateReport: (id: string, sections: Report["sections"]) =>
    request<Report>(`/api/reports/${id}`, { method: "PUT", body: JSON.stringify({ sections }) }),
  rewriteReport: (id: string, style: string, section?: string) =>
    request<Report>(`/api/reports/${id}/rewrite`, json({ style, section })),
  signReport: (id: string, signed_by: string) => request<Report>(`/api/reports/${id}/sign`, json({ signed_by })),
  fhir: (id: string) => request<Record<string, unknown>>(`/api/reports/${id}/fhir`),
  pdfUrl: (id: string) => `${API_URL}/api/reports/${id}/pdf`,
  review: (findingId: string, action: string, new_label?: string, comment?: string) =>
    request<{ status: string }>(`/api/findings/${findingId}/review`, json({ action, new_label, comment })),
  reviews: () => request<{ n: number; agreement_rate: number | null; items: unknown[] }>("/api/findings/reviews"),
  reviewsCsvUrl: () => `${API_URL}/api/findings/reviews/export`,
  compare: (prior_id: string, current_id: string, finding?: string) =>
    request<CompareResult>("/api/compare", json({ prior_id, current_id, finding })),
  enhance: (study_id: string, ops: Record<string, unknown>) =>
    request<EnhanceResult>("/api/enhance", json({ study_id, ops })),
  metrics: () => request<MetricsBundle>("/api/metrics"),
  modelCard: () => request<ModelCard>("/api/model-card"),
  audit: (action?: string) => request<AuditEntry[]>(`/api/audit${action ? `?action=${action}` : ""}`),
  chatHistory: (studyId: string) =>
    request<{ messages: (ChatAnswer & { role: string; content: string })[]; suggestions: string[] }>(
      `/api/copilot/${studyId}/history`,
    ),
  loadDemo: () => request<{ created: number; skipped: number }>("/api/demo/load", { method: "POST" }),
  federatedLast: () => request<Record<string, unknown>>("/api/federated/last"),
};

/** Upload with progress (XHR gives upload progress events; fetch does not). */
export function uploadStudy(
  file: File,
  fields: Record<string, string | undefined>,
  onProgress: (p: number) => void,
): Promise<Study> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    Object.entries(fields).forEach(([k, v]) => v && fd.append(k, v));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/studies`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new ApiError(xhr.status, body?.error?.code ?? "upload_failed", body?.error?.message ?? "Upload failed"));
      } catch {
        reject(new ApiError(xhr.status, "upload_failed", "Upload failed"));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network", "Network error — is the backend running?"));
    xhr.send(fd);
  });
}

/** POST + parse a Server-Sent Events stream. */
export async function streamSSE(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, "stream_failed", "Stream failed");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      try {
        onEvent(event, JSON.parse(data.join("\n")));
      } catch {
        onEvent(event, data.join("\n"));
      }
    }
  }
}

export interface Dashboard {
  total_studies: number;
  analyzed: number;
  this_week: number;
  critical: number;
  urgent: number;
  levels: Record<string, number>;
  findings_frequency: [string, number][];
  modalities: Record<string, number>;
  analysis_times: { id: string; ms: number; at: string | null; modality: string | null }[];
  mean_analysis_ms: number | null;
  reviews: number;
  agreement_rate: number | null;
  ood_count: number;
  recent_activity: AuditEntry[];
}

export interface AuditEntry {
  id: number;
  ts: string;
  actor: string;
  action: string;
  entity: string;
  entity_id: string | null;
  details: Record<string, unknown>;
}

export interface CompareResult {
  prior: string;
  current: string;
  deltas: { label: string; key: string; prior: number; current: number; delta: number; status: string; trend: string }[];
  ctr: { prior: number; current: number; delta: number } | null;
  triage: { prior: { level?: string; score?: number }; current: { level?: string; score?: number } };
  summary: string;
  heatmap_diff?: { finding: string; url?: string; error?: string };
  prior_study: StudySummary;
  current_study: StudySummary;
}

export interface EnhanceResult {
  image: string;
  before: string;
  metrics: { psnr: number; ssim: number; psnr_noisy?: number; ssim_noisy?: number };
  steps: string[];
  denoiser_model: boolean;
  ms: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyJson = any;

export interface MetricsBundle {
  models: Record<string, Record<string, AnyJson>>;
  federated: AnyJson;
  datasets: { key: string; title: string; license: string; citation: string; url: string; profiles: string[] }[];
  dataset_sizes: Record<string, Record<string, number>>;
}

export interface ModelCard {
  triage: { weights: Record<string, number>; levels: Record<string, number> };
  thresholds: Record<string, number>;
  temperatures: Record<string, number[]>;
  labels: string[];
  energy_threshold: number | null;
  chest_profile: string | null;
  headline: Record<string, AnyJson>;
}
