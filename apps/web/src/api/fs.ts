export type FsRoot = {
  label: string;
  path: string;
};

export type FsRootsResponse = {
  configured: boolean;
  roots: FsRoot[];
  invalidRoots: string[];
};

export type FsListItem = {
  name: string;
  path: string;
  isDir: boolean;
  isSymlink: boolean;
  mtimeMs: number | null;
};

export type FsListResponse = {
  path: string;
  rootPath: string;
  parentPath: string | null;
  page: number;
  pageSize: number;
  total: number;
  sort: "name" | "mtime";
  order: "asc" | "desc";
  includeHidden: boolean;
  dirsOnly: boolean;
  items: FsListItem[];
};

export async function fetchFsRoots(signal?: AbortSignal): Promise<FsRootsResponse> {
  const resp = await fetch("/api/fs/roots", { headers: { Accept: "application/json" }, signal });
  if (!resp.ok) {
    const detail = await safeDetail(resp);
    throw new Error(`FS roots failed: ${resp.status} ${resp.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return (await resp.json()) as FsRootsResponse;
}

export async function fetchFsList(
  params: {
    path: string;
    page: number;
    pageSize: number;
    sort: "name" | "mtime";
    order: "asc" | "desc";
    includeHidden: boolean;
    dirsOnly: boolean;
  },
  signal?: AbortSignal
): Promise<FsListResponse> {
  const search = new URLSearchParams({
    path: params.path,
    page: String(params.page),
    pageSize: String(params.pageSize),
    sort: params.sort,
    order: params.order,
    includeHidden: params.includeHidden ? "true" : "false",
    dirsOnly: params.dirsOnly ? "true" : "false"
  });

  const resp = await fetch(`/api/fs/list?${search.toString()}`, { headers: { Accept: "application/json" }, signal });
  if (!resp.ok) {
    const detail = await safeDetail(resp);
    throw new Error(`FS list failed: ${resp.status} ${resp.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return (await resp.json()) as FsListResponse;
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
