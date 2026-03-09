import { getClientId } from "../lib/clientId";

export type JobStatus = "running" | "completed" | "failed" | "halted";

export type JobDataset = {
  source_dir: string;
  save_to: string;
};

export type JobRoi = {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  z0: number;
  z1: number;
};

export type JobSummary = {
  jobId: string;
  type: string;
  status: JobStatus;
  createdAtMs: number;
  finishedAtMs: number | null;
  processed: number;
  total: number;
  error: string | null;
  current: string | null;
  label?: string | null;
  saveDir?: string | null;
  datasets?: JobDataset[];
  roi?: JobRoi | null;
  maskCval?: number | null;
  addedPadding?: number | null;
  invertLut?: boolean | null;
  copyMetadataRisky?: boolean | null;
  roundDownShapes?: boolean | null;
  overwrite?: boolean | null;
};

export type ListJobsResponse = {
  jobs: JobSummary[];
  nowMs: number;
};

function clientHeaders(): HeadersInit {
  return { "X-SpatialDINO-ClientId": getClientId() };
}

async function safeJson(resp: Response): Promise<unknown> {
  try {
    return (await resp.json()) as unknown;
  } catch {
    return null;
  }
}

export async function listJobs(options?: { signal?: AbortSignal }): Promise<ListJobsResponse> {
  const resp = await fetch("/api/jobs/list", {
    method: "GET",
    headers: { Accept: "application/json", ...clientHeaders() },
    signal: options?.signal,
  });
  if (!resp.ok) {
    throw new Error(`List jobs failed: ${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as ListJobsResponse;
}

export async function clearFinishedJobs(options?: { signal?: AbortSignal }): Promise<void> {
  const resp = await fetch("/api/jobs/clear", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json", ...clientHeaders() },
    body: JSON.stringify({ keep_running: true }),
    signal: options?.signal,
  });
  if (!resp.ok) {
    throw new Error(`Clear jobs failed: ${resp.status} ${resp.statusText}`);
  }
}

export async function cancelJob(jobId: string, options?: { signal?: AbortSignal }): Promise<void> {
  const resp = await fetch("/api/jobs/cancel", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json", ...clientHeaders() },
    body: JSON.stringify({ job_id: jobId }),
    signal: options?.signal,
  });
  if (!resp.ok) {
    const json = await safeJson(resp);
    throw new Error(`Cancel job failed: ${resp.status} ${resp.statusText}${json ? `: ${JSON.stringify(json)}` : ""}`);
  }
}

export async function removeJob(jobId: string, options?: { signal?: AbortSignal }): Promise<void> {
  const resp = await fetch("/api/jobs/remove", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json", ...clientHeaders() },
    body: JSON.stringify({ job_id: jobId }),
    signal: options?.signal,
  });
  if (!resp.ok) {
    const json = await safeJson(resp);
    throw new Error(`Remove job failed: ${resp.status} ${resp.statusText}${json ? `: ${JSON.stringify(json)}` : ""}`);
  }
}
