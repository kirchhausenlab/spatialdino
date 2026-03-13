import { getClientId } from "../lib/clientId";

export type PublicDataDataset = {
  name: string;
};

export type DataOptionsResponse = {
  manifestUrl: string;
  downloadRoot: string;
  datasets: PublicDataDataset[];
};

export type DataDownloadRequest = {
  datasets: string[];
  existing_mode?: "skip" | "overwrite";
};

export type DataDownloadSubmittedResponse = {
  submitted: true;
  jobId: string;
  message: string;
};

export type DataDownloadOverwritePromptResponse = {
  submitted: false;
  valid: true;
  requiresOverwriteConfirmation: true;
  message: string;
  existingDatasetCount: number;
  existingDatasetNames: string[];
  existingDatasetPaths: string[];
};

export type DataDownloadInvalidResponse = {
  submitted: false;
  valid: false;
  reasonCode: string;
  message: string;
};

export type DataDownloadResponse =
  | DataDownloadSubmittedResponse
  | DataDownloadOverwritePromptResponse
  | DataDownloadInvalidResponse;

function clientHeaders(): HeadersInit {
  return { "X-SpatialDINO-ClientId": getClientId() };
}

export async function fetchDataOptions(signal?: AbortSignal): Promise<DataOptionsResponse> {
  const resp = await fetch("/api/data/options", {
    method: "GET",
    headers: { Accept: "application/json", ...clientHeaders() },
    signal,
  });
  if (!resp.ok) {
    const detail = await safeDetail(resp);
    throw new Error(`Data options failed: ${resp.status} ${resp.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return (await resp.json()) as DataOptionsResponse;
}

export async function submitDataDownload(
  payload: DataDownloadRequest,
  signal?: AbortSignal
): Promise<DataDownloadResponse> {
  const resp = await fetch("/api/data/download", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...clientHeaders(),
    },
    body: JSON.stringify(payload),
    signal,
  });
  if (!resp.ok) {
    const detail = await safeDetail(resp);
    throw new Error(`Data download failed: ${resp.status} ${resp.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return (await resp.json()) as DataDownloadResponse;
}

async function safeDetail(resp: Response): Promise<string | null> {
  try {
    const json = (await resp.json()) as unknown;
    if (json && typeof json === "object" && "detail" in json) {
      const detail = (json as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
    return JSON.stringify(json);
  } catch {
    try {
      const text = (await resp.text()).trim();
      return text.length > 0 ? text : null;
    } catch {
      return null;
    }
  }
}
