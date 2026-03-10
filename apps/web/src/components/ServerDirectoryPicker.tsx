import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchFsList, fetchFsMkdir } from "../api/fs";
import type { FsListResponse } from "../api/fs";
import Modal from "./Modal";

type CacheEntry<T> = { atMs: number; value: T };
const VISIBLE_ROWS = 10;
const FETCH_PAGE_SIZE = 200;
const MAX_BOOKMARKED_DIRS = 10;
const BOOKMARKS_STORAGE_KEY = "spatialdino.picker.bookmarks.v1";
const BOOKMARKS_SYNC_EVENT = "spatialdino:picker-bookmarks-updated";
let lastVisitedPickerPath: string | null = null;

export function getServerDirectoryPickerCurrentPath(): string {
  return lastVisitedPickerPath ?? "/";
}

type NavigateResult = { ok: true } | { ok: false; error: string };

function cacheKey(parts: Record<string, unknown>): string {
  const sorted = Object.fromEntries(Object.entries(parts).sort(([a], [b]) => a.localeCompare(b)));
  return JSON.stringify(sorted);
}

function formatMtime(ms: number | null): string {
  if (!ms) return "—";
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString();
}

function sanitizeBookmarkedPath(raw: string): string | null {
  const path = raw.trim();
  if (!path.startsWith("/")) return null;
  if (path.includes("\x00")) return null;
  return path;
}

function normalizeBookmarkedDirs(paths: readonly string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const rawPath of paths) {
    const safePath = sanitizeBookmarkedPath(rawPath);
    if (!safePath || seen.has(safePath)) continue;
    seen.add(safePath);
    out.push(safePath);
    if (out.length >= MAX_BOOKMARKED_DIRS) break;
  }
  return out;
}

function readStoredBookmarkedDirs(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(BOOKMARKS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return normalizeBookmarkedDirs(parsed.filter((v): v is string => typeof v === "string"));
  } catch (error) {
    console.warn("[spatialDINO] Unable to read bookmarked directories from localStorage:", error);
    return [];
  }
}

function writeStoredBookmarkedDirs(paths: readonly string[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(BOOKMARKS_STORAGE_KEY, JSON.stringify(paths));
  } catch (error) {
    console.warn("[spatialDINO] Unable to persist bookmarked directories to localStorage:", error);
  }
}

function isInvalidBookmarkedDirError(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("path must be absolute") ||
    normalized.includes("invalid path") ||
    normalized.includes("path is outside configured roots") ||
    normalized.includes("path does not exist") ||
    normalized.includes("path is not a directory")
  );
}

function extractErrorDetail(message: string): string {
  const idx = message.lastIndexOf(": ");
  return idx >= 0 ? message.slice(idx + 2) : message;
}

function isCreateWarningMessage(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("already exists") ||
    normalized.includes("folder name") ||
    normalized.includes("hidden folder names") ||
    normalized.includes("invalid folder name")
  );
}

function StarIcon({ filled }: { filled: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      role="img"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M12 2.6L14.7 8l5.9.9-4.3 4.2 1 5.9-5.3-2.8-5.3 2.8 1-5.9-4.3-4.2L9.3 8 12 2.6Z"
        fill={filled ? "currentColor" : "none"}
        stroke={filled ? "none" : "currentColor"}
        strokeWidth="1.9"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function ServerDirectoryPicker({
  open,
  title,
  initialPath,
  onClose,
  onSelect
}: {
  open: boolean;
  title?: string;
  initialPath?: string | null;
  onClose: () => void;
  onSelect: (path: string) => void;
}) {
  const listCache = useRef<Map<string, CacheEntry<FsListResponse>>>(new Map());
  const cacheTtlMs = 2000;

  const resolveInitialPath = useCallback((nextInitialPath: string | null | undefined) => {
    return nextInitialPath ?? lastVisitedPickerPath ?? "/";
  }, []);

  const [path, setPath] = useState<string>(() => resolveInitialPath(initialPath));
  const [pathInput, setPathInput] = useState<string>(() => resolveInitialPath(initialPath));
  const [sort, setSort] = useState<"name" | "mtime">("name");
  const [order, setOrder] = useState<"asc" | "desc">("asc");

  const [listState, setListState] = useState<{
    loading: boolean;
    error: string | null;
    data: FsListResponse | null;
    items: FsListResponse["items"];
  }>({ loading: false, error: null, data: null, items: [] });

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [bookmarkedDirs, setBookmarkedDirs] = useState<string[]>(() => readStoredBookmarkedDirs());
  const [bookmarksOpen, setBookmarksOpen] = useState(false);
  const [newDirName, setNewDirName] = useState<string | null>(null);
  const [newDirBusy, setNewDirBusy] = useState(false);
  const [newDirMessage, setNewDirMessage] = useState<{ tone: "warning" | "error"; text: string } | null>(null);

  const loadedPagesRef = useRef<Set<number>>(new Set());
  const loadMoreRef = useRef<{
    inFlight: boolean;
    requestedAtMs: number;
  }>({ inFlight: false, requestedAtMs: 0 });
  const bookmarkMenuRef = useRef<HTMLDivElement | null>(null);
  const newDirInputRef = useRef<HTMLInputElement | null>(null);

  const commitBookmarkedDirs = useCallback((updater: (prev: string[]) => string[]) => {
    setBookmarkedDirs((prev) => {
      const next = normalizeBookmarkedDirs(updater([...prev]));
      writeStoredBookmarkedDirs(next);
      window.dispatchEvent(new CustomEvent<string[]>(BOOKMARKS_SYNC_EVENT, { detail: next }));
      return next;
    });
  }, []);

  useEffect(() => {
    if (!open) return;
    const next = resolveInitialPath(initialPath);
    setPath(next);
    setPathInput(next);
    setSelectedPath(null);
  }, [open, initialPath, resolveInitialPath]);

  useEffect(() => {
    if (!open) return;
    lastVisitedPickerPath = path;
  }, [open, path]);

  useEffect(() => {
    if (!open) return;
    setSelectedPath(null);
  }, [open, path]);

  useEffect(() => {
    if (!open) {
      setBookmarksOpen(false);
      return;
    }
    setBookmarkedDirs(readStoredBookmarkedDirs());
  }, [open]);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== BOOKMARKS_STORAGE_KEY) return;
      setBookmarkedDirs(readStoredBookmarkedDirs());
    };
    const onInternalSync = (event: Event) => {
      const detail = (event as CustomEvent<string[]>).detail;
      if (!Array.isArray(detail)) return;
      setBookmarkedDirs(normalizeBookmarkedDirs(detail));
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(BOOKMARKS_SYNC_EVENT, onInternalSync);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(BOOKMARKS_SYNC_EVENT, onInternalSync);
    };
  }, []);

  useEffect(() => {
    if (!bookmarksOpen) return;
    const onMouseDown = (event: MouseEvent) => {
      const root = bookmarkMenuRef.current;
      if (!root) return;
      if (event.target instanceof Node && root.contains(event.target)) return;
      setBookmarksOpen(false);
    };
    window.addEventListener("mousedown", onMouseDown);
    return () => window.removeEventListener("mousedown", onMouseDown);
  }, [bookmarksOpen]);

  const queryKeyBase = useMemo(() => {
    return cacheKey({ path, sort, order, includeHidden: false, dirsOnly: true });
  }, [path, sort, order]);

  const clearListCache = useCallback(() => {
    listCache.current.clear();
    loadedPagesRef.current = new Set();
    loadMoreRef.current = { inFlight: false, requestedAtMs: 0 };
  }, []);

  const resetAndLoadFirstPage = useMemo(() => {
    return () => {
      loadedPagesRef.current = new Set();
      loadMoreRef.current = { inFlight: false, requestedAtMs: 0 };
      setListState({ loading: true, error: null, data: null, items: [] });
    };
  }, []);

  const navigateTo = useCallback(
    async (raw: string): Promise<NavigateResult> => {
      const target = raw.trim();
      if (!target.startsWith("/")) {
        const message = "Path must be absolute.";
        setListState((prev) => ({ ...prev, loading: false, error: message }));
        return { ok: false, error: message };
      }
      if (target.includes("\x00")) {
        const message = "Invalid path.";
        setListState((prev) => ({ ...prev, loading: false, error: message }));
        return { ok: false, error: message };
      }

      setListState((prev) => ({ ...prev, loading: true, error: null }));
      const page = 1;

      try {
        const data = await fetchFsList({ path: target, page, pageSize: FETCH_PAGE_SIZE, sort, order, includeHidden: false, dirsOnly: true });
        const canonicalKeyBase = cacheKey({ path: data.path, sort, order, includeHidden: false, dirsOnly: true });
        const canonicalQueryKey = cacheKey({ base: canonicalKeyBase, page, pageSize: FETCH_PAGE_SIZE });
        listCache.current.set(canonicalQueryKey, { atMs: Date.now(), value: data });
        loadedPagesRef.current = new Set([1]);
        loadMoreRef.current = { inFlight: false, requestedAtMs: 0 };
        setPath(data.path);
        setPathInput(data.path);
        setSelectedPath(null);
        setListState({ loading: false, error: null, data, items: data.items });
        return { ok: true };
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        setListState((prev) => ({ ...prev, loading: false, error: message }));
        return { ok: false, error: message };
      }
    },
    [order, sort]
  );

  useEffect(() => {
    if (!open) return;
    if (!queryKeyBase) return;

    const controller = new AbortController();
    const page = 1;
    const queryKey = cacheKey({ base: queryKeyBase, page, pageSize: FETCH_PAGE_SIZE });

    const now = Date.now();
    const cached = listCache.current.get(queryKey);
    if (cached && now - cached.atMs < cacheTtlMs) {
      loadedPagesRef.current.add(page);
      setListState({ loading: false, error: null, data: cached.value, items: cached.value.items });
      return () => controller.abort();
    }

    resetAndLoadFirstPage();

    void (async () => {
      try {
        const data = await fetchFsList(
          { path, page, pageSize: FETCH_PAGE_SIZE, sort, order, includeHidden: false, dirsOnly: true },
          controller.signal
        );
        const canonicalKeyBase = cacheKey({ path: data.path, sort, order, includeHidden: false, dirsOnly: true });
        const canonicalQueryKey = cacheKey({ base: canonicalKeyBase, page, pageSize: FETCH_PAGE_SIZE });
        listCache.current.set(canonicalQueryKey, { atMs: Date.now(), value: data });
        if (data.path !== path) {
          setPath(data.path);
          setPathInput(data.path);
        }
        loadedPagesRef.current.add(page);
        setListState({ loading: false, error: null, data, items: data.items });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        setListState({ loading: false, error: message, data: null, items: [] });
      }
    })();

    return () => controller.abort();
  }, [open, path, sort, order, queryKeyBase, resetAndLoadFirstPage, cacheTtlMs]);

  const canLoadMore = useMemo(() => {
    const total = listState.data?.total ?? 0;
    return listState.items.length < total;
  }, [listState.data, listState.items.length]);

  const loadNextPage = useMemo(() => {
    return async () => {
      if (!open) return;
      if (!queryKeyBase) return;
      if (!listState.data) return;
      if (!canLoadMore) return;
      if (loadMoreRef.current.inFlight) return;

      const nextPage = Math.floor(listState.items.length / FETCH_PAGE_SIZE) + 1;
      if (loadedPagesRef.current.has(nextPage)) return;

      loadMoreRef.current.inFlight = true;
      loadMoreRef.current.requestedAtMs = Date.now();

      const queryKey = cacheKey({ base: queryKeyBase, page: nextPage, pageSize: FETCH_PAGE_SIZE });

      try {
        const now = Date.now();
        const cached = listCache.current.get(queryKey);
        const data =
          cached && now - cached.atMs < cacheTtlMs
            ? cached.value
            : await fetchFsList({
                path,
                page: nextPage,
                pageSize: FETCH_PAGE_SIZE,
                sort,
                order,
                includeHidden: false,
                dirsOnly: true
              });

        listCache.current.set(queryKey, { atMs: Date.now(), value: data });
        loadedPagesRef.current.add(nextPage);
        setListState((prev) => ({
          loading: false,
          error: null,
          data,
          items: [...prev.items, ...data.items]
        }));
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        setListState((prev) => ({ ...prev, loading: false, error: message }));
      } finally {
        loadMoreRef.current.inFlight = false;
      }
    };
  }, [open, path, sort, order, queryKeyBase, listState.data, listState.items.length, canLoadMore]);

  const currentPath = listState.data?.path ?? path ?? "";
  const currentBookmarkPath = sanitizeBookmarkedPath(currentPath);
  const bookmarkTargetPath = currentBookmarkPath ?? sanitizeBookmarkedPath(pathInput);
  const isCurrentBookmarked = Boolean(bookmarkTargetPath && bookmarkedDirs.includes(bookmarkTargetPath));
  const isCreatingDir = newDirName !== null;
  const selectedLabel = selectedPath ?? "";
  const showEmptyRow = Boolean(listState.data && !listState.loading && !listState.error && !isCreatingDir && listState.items.length === 0);
  const visibleRowCount = listState.items.length + (isCreatingDir ? 1 : 0) + (showEmptyRow ? 1 : 0);
  const placeholderCount = Math.max(0, VISIBLE_ROWS - Math.min(VISIBLE_ROWS, visibleRowCount));
  const placeholders = useMemo(() => Array.from({ length: placeholderCount }), [placeholderCount]);

  const navigateToBookmarkedDir = useCallback(
    async (bookmarkedPath: string) => {
      if (isCreatingDir) return;
      const result = await navigateTo(bookmarkedPath);
      setBookmarksOpen(false);
      if (!result.ok && isInvalidBookmarkedDirError(result.error)) {
        commitBookmarkedDirs((prev) => prev.filter((entry) => entry !== bookmarkedPath));
      }
    },
    [commitBookmarkedDirs, isCreatingDir, navigateTo]
  );

  useEffect(() => {
    if (!open) return;
    setBookmarksOpen(false);
  }, [open, currentPath]);

  useEffect(() => {
    if (!open) {
      setNewDirName(null);
      setNewDirBusy(false);
      setNewDirMessage(null);
      return;
    }
    setNewDirName(null);
    setNewDirBusy(false);
    setNewDirMessage(null);
  }, [open, currentPath]);

  useEffect(() => {
    if (!isCreatingDir) return;
    const t = window.setTimeout(() => {
      newDirInputRef.current?.focus();
      newDirInputRef.current?.select();
    }, 0);
    return () => window.clearTimeout(t);
  }, [isCreatingDir]);

  const commitNewDir = useCallback(async () => {
    if (newDirName === null || newDirBusy) return;
    if (!listState.data) return;

    if (!newDirName.trim()) {
      setNewDirName(null);
      setNewDirBusy(false);
      setNewDirMessage(null);
      return;
    }

    const parentPath = listState.data.path;
    setNewDirBusy(true);
    setNewDirMessage(null);

    try {
      const created = await fetchFsMkdir({ parentPath, name: newDirName });
      clearListCache();
      const result = await navigateTo(parentPath);
      setSelectedPath(created.path);
      setNewDirName(null);
      setNewDirBusy(false);
      if (!result.ok) {
        setNewDirMessage({ tone: "error", text: result.error });
      }
    } catch (error) {
      const text = extractErrorDetail(error instanceof Error ? error.message : "Unable to create folder.");
      setNewDirBusy(false);
      setNewDirMessage({ tone: isCreateWarningMessage(text) ? "warning" : "error", text });
      window.setTimeout(() => {
        newDirInputRef.current?.focus();
        newDirInputRef.current?.select();
      }, 0);
    }
  }, [clearListCache, listState.data, navigateTo, newDirBusy, newDirName]);

  const cancelNewDir = useCallback(() => {
    if (newDirBusy) return;
    setNewDirName(null);
    setNewDirMessage(null);
  }, [newDirBusy]);
  const handleClose = useCallback(() => {
    if (newDirBusy) return;
    onClose();
  }, [newDirBusy, onClose]);

  return (
    <Modal
      open={open}
      title={title ?? "Choose a directory"}
      onClose={handleClose}
      panelClassName="pickerModalPanel"
      bodyClassName="pickerModalBody"
      footer={
        <div className="pickerFooter">
          <div className="pickerFooterPath">
            <span className="pickerFooterPathLabel">Selected:</span>
            <span className="pickerFooterPathValue" title={selectedLabel}>
              {selectedPath ?? "—"}
            </span>
          </div>
          <div className="pickerFooterActions">
            <button
              type="button"
              className="pickerSecondaryButton"
              disabled={!listState.data || isCreatingDir || newDirBusy}
              onClick={() => {
                if (!listState.data || isCreatingDir || newDirBusy) return;
                setSelectedPath(null);
                setNewDirName("");
                setNewDirMessage(null);
              }}
            >
              {newDirBusy ? "Creating..." : "New folder"}
            </button>
            <button
              type="button"
              className="pickerSecondaryButton"
              data-picker-create-cancel="true"
              onClick={handleClose}
              disabled={newDirBusy}
            >
              Cancel
            </button>
            <button
              type="button"
              className="pickerPrimaryButton"
              disabled={!selectedPath}
              onClick={() => {
                if (selectedPath) onSelect(selectedPath);
              }}
            >
              Use this directory
            </button>
          </div>
        </div>
      }
    >
      <div className="pickerToolbar">
        <div className="pickerToolbarLeft">
          <button
            type="button"
            className="sidebarActionButton"
            disabled={!listState.data?.parentPath}
            onClick={() => {
              if (isCreatingDir) return;
              const parent = listState.data?.parentPath;
              if (!parent) return;
              void navigateTo(parent);
            }}
            aria-label="Up"
            title="Up"
          >
            ↑
          </button>
          <form
            className="pickerPathForm"
            onSubmit={(e) => {
              e.preventDefault();
              void navigateTo(pathInput);
            }}
          >
            <div className="pickerPathInputWrap" ref={bookmarkMenuRef}>
              <input
                className="pickerPathInput"
                value={pathInput}
                onChange={(e) => setPathInput(e.target.value)}
                spellCheck={false}
                inputMode="text"
                aria-label="Current directory"
                title={currentPath || ""}
                placeholder="/path/to/directory"
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    e.preventDefault();
                    setPathInput(currentPath || "/");
                  }
                }}
              />
              <button
                type="button"
                className="pickerPathDropdownButton"
                aria-label="Bookmarked directories"
                aria-haspopup="menu"
                aria-expanded={bookmarksOpen}
                title="Bookmarked directories"
                onClick={() => {
                  if (isCreatingDir) return;
                  setBookmarksOpen((prev) => !prev);
                }}
              >
                ▾
              </button>
              {bookmarksOpen ? (
                <div className="pickerBookmarksMenu" role="menu" aria-label="Bookmarked directories">
                  {bookmarkedDirs.map((bookmarkedPath) => (
                    <button
                      key={bookmarkedPath}
                      type="button"
                      role="menuitem"
                      className={
                        bookmarkedPath === currentBookmarkPath
                          ? "pickerBookmarksMenuItem isCurrent"
                          : "pickerBookmarksMenuItem"
                      }
                      title={bookmarkedPath}
                      onClick={() => {
                        void navigateToBookmarkedDir(bookmarkedPath);
                      }}
                    >
                      {bookmarkedPath}
                    </button>
                  ))}
                  <button
                    type="button"
                    role="menuitem"
                    className="pickerBookmarksMenuItem pickerBookmarksMenuClear"
                    title="Clear bookmarked directories"
                    onClick={() => {
                      if (isCreatingDir) return;
                      commitBookmarkedDirs(() => []);
                      setBookmarksOpen(false);
                    }}
                  >
                    Clear Bookmarks
                  </button>
                </div>
              ) : null}
            </div>
          </form>
        </div>

        <button
          type="button"
          className={isCurrentBookmarked ? "pickerBookmarkToggle isActive" : "pickerBookmarkToggle"}
          aria-label={isCurrentBookmarked ? "Remove directory bookmark" : "Bookmark current directory"}
          aria-pressed={isCurrentBookmarked}
          title={isCurrentBookmarked ? "Remove from Bookmarks" : "Add to Bookmarks"}
          disabled={!bookmarkTargetPath}
          onClick={() => {
            if (!bookmarkTargetPath) return;
            commitBookmarkedDirs((prev) => {
              if (prev.includes(bookmarkTargetPath)) {
                return prev.filter((entry) => entry !== bookmarkTargetPath);
              }
              return [bookmarkTargetPath, ...prev].slice(0, MAX_BOOKMARKED_DIRS);
            });
          }}
        >
          <StarIcon filled={isCurrentBookmarked} />
        </button>

        <div className="pickerToolbarRight">
          <div className="pickerSortLabel">Sort by</div>
          <select
            className="pickerSelect"
            value={sort}
            onChange={(e) => {
              if (isCreatingDir) return;
              setSort(e.target.value as "name" | "mtime");
            }}
            aria-label="Sort by"
          >
            <option value="name">Name</option>
            <option value="mtime">Modified</option>
          </select>
          <button
            type="button"
            className="pickerOrderButton"
            aria-label="Toggle sort order"
            onClick={() => {
              if (isCreatingDir) return;
              setOrder((prev) => (prev === "asc" ? "desc" : "asc"));
            }}
          >
            {order === "asc" ? "↑↓" : "↓↑"}
          </button>
        </div>
      </div>

      {newDirMessage ? (
        <div
          className={newDirMessage.tone === "warning" ? "sidebarWarning" : "sidebarError"}
          role="alert"
          aria-live="polite"
        >
          {newDirMessage.text}
        </div>
      ) : null}
      {listState.error ? <div className="sidebarError">{listState.error}</div> : null}

      {listState.data ? (
        <>
          <div
            className="pickerTableWrap"
            role="table"
            aria-label="Directories"
            onScroll={(e) => {
              if (listState.loading) return;
              if (!canLoadMore) return;
              const el = e.currentTarget;
              const thresholdPx = 60;
              if (el.scrollTop + el.clientHeight >= el.scrollHeight - thresholdPx) {
                void loadNextPage();
              }
            }}
          >
            <div className="pickerTableHeader" role="row">
              <div className="pickerTh pickerNameCol" role="columnheader">
                Name
              </div>
              <div className="pickerTh pickerModifiedCol" role="columnheader">
                Modified
              </div>
            </div>
            {isCreatingDir ? (
              <div className="pickerRow pickerCreateRow" role="row">
                <div className="pickerTd pickerNameCol" role="cell">
                  <input
                    ref={newDirInputRef}
                    className="pickerInlineInput"
                    value={newDirName ?? ""}
                    onChange={(e) => {
                      setNewDirName(e.target.value);
                      if (newDirMessage) setNewDirMessage(null);
                    }}
                    spellCheck={false}
                    inputMode="text"
                    aria-label="New folder name"
                    placeholder="New folder"
                    disabled={newDirBusy}
                    onBlur={(e) => {
                      const nextFocused = e.relatedTarget;
                      if (
                        nextFocused instanceof HTMLElement &&
                        (nextFocused.closest("[data-picker-create-cancel='true']") ||
                          nextFocused.closest("[data-modal-close='true']"))
                      ) {
                        cancelNewDir();
                        return;
                      }
                      void commitNewDir();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void commitNewDir();
                        return;
                      }
                      if (e.key === "Escape") {
                        e.preventDefault();
                        cancelNewDir();
                      }
                    }}
                  />
                </div>
                <div className="pickerTd pickerModifiedCol pickerCreateMeta" role="cell">
                  {newDirBusy ? "Creating..." : "—"}
                </div>
              </div>
            ) : null}
            {showEmptyRow ? (
              <div className="pickerRow isPlaceholder" role="row">
                <div className="pickerTd pickerNameCol pickerEmptyCell" role="cell">
                  No directories.
                </div>
                <div className="pickerTd pickerModifiedCol" role="cell" />
              </div>
            ) : null}

            {listState.items.map((item) => (
              <div
                key={item.path}
                className={item.path === selectedPath ? "pickerRow isSelected" : "pickerRow"}
                role="row"
              >
                <div className="pickerTd pickerNameCol" role="cell">
                  <button
                    type="button"
                    className="pickerEntryButton"
                    onClick={() => {
                      if (isCreatingDir) return;
                      setSelectedPath(item.path);
                    }}
                    onDoubleClick={() => {
                      if (isCreatingDir) return;
                      void navigateTo(item.path);
                    }}
                  >
                    {item.name}
                    {item.isSymlink ? <span className="pickerSymlink"> ↪</span> : null}
                  </button>
                </div>
                <div className="pickerTd pickerModifiedCol" role="cell">
                  {formatMtime(item.mtimeMs)}
                </div>
              </div>
            ))}
            {placeholders.map((_, idx) => (
              <div key={`placeholder-${idx}`} className="pickerRow isPlaceholder" role="row" aria-hidden="true">
                <div className="pickerTd pickerNameCol" role="cell" />
                <div className="pickerTd pickerModifiedCol" role="cell" />
              </div>
            ))}
          </div>
        </>
      ) : null}
    </Modal>
  );
}
