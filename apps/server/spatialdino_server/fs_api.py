from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from spatialdino_server.fs_roots import _configured_fs_roots_from_env

router = APIRouter(prefix="/fs")

SortKey = Literal["name", "mtime"]
SortOrder = Literal["asc", "desc"]


class CreateDirRequest(BaseModel):
    parent_path: str = Field(..., min_length=1, alias="parentPath")
    name: str = Field(..., min_length=0)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"Invalid boolean: {value!r}")


def _parse_int(value: str | None, *, default: int, min_value: int, max_value: int, name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {name}: {value!r}") from exc
    if parsed < min_value or parsed > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {name}: {parsed} (must be {min_value}..{max_value})",
        )
    return parsed


def _parse_sort(value: str | None, default: SortKey) -> SortKey:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"name", "mtime"}:
        return v  # type: ignore[return-value]
    raise HTTPException(status_code=400, detail=f"Invalid sort: {value!r}")


def _parse_order(value: str | None, default: SortOrder) -> SortOrder:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"asc", "desc"}:
        return v  # type: ignore[return-value]
    raise HTTPException(status_code=400, detail=f"Invalid order: {value!r}")


@dataclass(frozen=True)
class RootInfo:
    label: str
    real_path: Path


def _get_configured_roots() -> tuple[list[RootInfo], list[str]]:
    configured = _configured_fs_roots_from_env()
    roots = [RootInfo(label=str(path), real_path=path) for path in configured.roots]
    return roots, configured.invalid


def _find_root_for_path(path: Path, roots: list[RootInfo]) -> RootInfo | None:
    for root in roots:
        try:
            path.relative_to(root.real_path)
        except ValueError:
            continue
        return root
    return None


def _require_allowed_dir(requested: str, roots: list[RootInfo]) -> tuple[Path, RootInfo]:
    if not roots:
        raise HTTPException(
            status_code=503,
            detail="No filesystem roots configured. Set SPATIALDINO_FS_ROOTS (os.pathsep-separated).",
        )
    if "\x00" in requested:
        raise HTTPException(status_code=400, detail="Invalid path.")

    configured = Path(requested).expanduser()
    if not configured.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute.")

    real = Path(os.path.realpath(configured))
    root = _find_root_for_path(real, roots)
    if root is None:
        raise HTTPException(status_code=403, detail="Path is outside configured roots.")

    if not real.exists():
        raise HTTPException(status_code=404, detail="Path does not exist.")
    if not real.is_dir():
        raise HTTPException(status_code=404, detail="Path is not a directory.")

    return real, root


def _validate_dir_name(name: str) -> str:
    if "\x00" in name:
        raise HTTPException(status_code=400, detail="Invalid folder name.")

    if not name.strip():
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")

    if name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid folder name.")

    separators = {"/"}
    if os.sep:
        separators.add(os.sep)
    if os.altsep:
        separators.add(os.altsep)

    if any(separator in name for separator in separators if separator):
        raise HTTPException(status_code=400, detail="Folder name must not contain path separators.")

    if name.startswith("."):
        raise HTTPException(status_code=400, detail="Hidden folder names are not supported here.")

    return name


@dataclass(frozen=True)
class _CacheKey:
    path: str
    include_hidden: bool
    dirs_only: bool
    sort: SortKey


@dataclass(frozen=True)
class _CacheValue:
    created_at: float
    # For sort=name, mtimeMs may be None and computed only for the returned page slice.
    items: list[dict]


_CACHE_LOCK = threading.Lock()
_CACHE: dict[_CacheKey, _CacheValue] = {}
_CACHE_TTL_S = 2.0
_CACHE_MAX_KEYS = 128


def _cache_get(key: _CacheKey) -> _CacheValue | None:
    now = time.monotonic()
    with _CACHE_LOCK:
        value = _CACHE.get(key)
        if value is None:
            return None
        if now - value.created_at > _CACHE_TTL_S:
            _CACHE.pop(key, None)
            return None
        return value


def _cache_put(key: _CacheKey, value: _CacheValue) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX_KEYS:
            # Simple eviction: drop the oldest entry.
            oldest_key = min(_CACHE.items(), key=lambda kv: kv[1].created_at)[0]
            _CACHE.pop(oldest_key, None)
        _CACHE[key] = value


def _cache_invalidate_dir(dir_path: Path) -> None:
    path_text = str(dir_path)
    with _CACHE_LOCK:
        keys_to_drop = [key for key in _CACHE if key.path == path_text]
        for key in keys_to_drop:
            _CACHE.pop(key, None)


def _scan_children(
    *,
    dir_path: Path,
    root_path: Path,
    include_hidden: bool,
    dirs_only: bool,
    sort: SortKey,
) -> list[dict]:
    items: list[dict] = []
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                name = entry.name
                if not include_hidden and name.startswith("."):
                    continue

                is_symlink = entry.is_symlink()

                is_dir_no_follow = entry.is_dir(follow_symlinks=False)
                is_dir_follow = False
                if not is_dir_no_follow and is_symlink:
                    # Only follow symlinks when they resolve to a directory and remain within the root.
                    is_dir_follow = entry.is_dir(follow_symlinks=True)

                is_dir = is_dir_no_follow or is_dir_follow
                if dirs_only and not is_dir:
                    continue
                if is_symlink and is_dir_follow:
                    target_real = Path(os.path.realpath(entry.path))
                    try:
                        target_real.relative_to(root_path)
                    except ValueError:
                        continue

                item: dict = {
                    "name": name,
                    "path": entry.path,
                    "isDir": is_dir,
                    "isSymlink": is_symlink,
                    "mtimeMs": None,
                }

                if sort == "mtime":
                    try:
                        stat = entry.stat(follow_symlinks=False)
                        item["mtimeMs"] = int(stat.st_mtime * 1000)
                    except OSError:
                        item["mtimeMs"] = 0

                items.append(item)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Path does not exist.") from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=404, detail="Path is not a directory.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied.") from exc

    if sort == "name":
        items.sort(key=lambda i: (str(i["name"]).casefold(), str(i["name"])))
    else:
        items.sort(key=lambda i: (int(i["mtimeMs"] or 0), str(i["name"]).casefold(), str(i["name"])))
    return items


def _get_children_cached(
    *,
    dir_path: Path,
    root_path: Path,
    include_hidden: bool,
    dirs_only: bool,
    sort: SortKey,
) -> list[dict]:
    key = _CacheKey(
        path=str(dir_path),
        include_hidden=include_hidden,
        dirs_only=dirs_only,
        sort=sort,
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached.items

    items = _scan_children(
        dir_path=dir_path,
        root_path=root_path,
        include_hidden=include_hidden,
        dirs_only=dirs_only,
        sort=sort,
    )
    _cache_put(key, _CacheValue(created_at=time.monotonic(), items=items))
    return items


@router.get("/roots")
def fs_roots() -> dict:
    roots, invalid = _get_configured_roots()
    return {
        "configured": bool(roots),
        "roots": [{"label": r.label, "path": str(r.real_path)} for r in roots],
        "invalidRoots": invalid,
    }


@router.get("/list")
def fs_list(
    path: str,
    page: str | None = None,
    pageSize: str | None = None,
    sort: str | None = None,
    order: str | None = None,
    includeHidden: str | None = None,
    dirsOnly: str | None = None,
) -> dict:
    roots, _invalid = _get_configured_roots()
    include_hidden_b = _parse_bool(includeHidden, default=False)
    dirs_only_b = _parse_bool(dirsOnly, default=True)
    sort_key = _parse_sort(sort, default="name")
    sort_order = _parse_order(order, default="asc")
    page_size = _parse_int(pageSize, default=25, min_value=1, max_value=500, name="pageSize")
    page_num = _parse_int(page, default=1, min_value=1, max_value=10_000_000, name="page")

    dir_path, root = _require_allowed_dir(path, roots)
    items_sorted = _get_children_cached(
        dir_path=dir_path,
        root_path=root.real_path,
        include_hidden=include_hidden_b,
        dirs_only=dirs_only_b,
        sort=sort_key,
    )

    total = len(items_sorted)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page_num > total_pages:
        raise HTTPException(status_code=400, detail=f"Page out of range (max {total_pages}).")

    start = (page_num - 1) * page_size
    end = min(start + page_size, total)

    if sort_order == "asc":
        page_items = [dict(item) for item in items_sorted[start:end]]
    else:
        # Avoid reversing the full list for large directories.
        start_desc = max(0, total - end)
        end_desc = total - start
        page_items = [dict(item) for item in reversed(items_sorted[start_desc:end_desc])]

    # Avoid eager stat() calls when sorting by name; only compute mtime for returned items.
    if sort_key == "name":
        for item in page_items:
            if item.get("mtimeMs") is not None:
                continue
            try:
                stat = os.stat(item["path"], follow_symlinks=False)
                item["mtimeMs"] = int(stat.st_mtime * 1000)
            except OSError:
                item["mtimeMs"] = 0

    parent_path: str | None = None
    if dir_path != root.real_path:
        candidate = dir_path.parent
        try:
            candidate.relative_to(root.real_path)
        except ValueError:
            parent_path = None
        else:
            parent_path = str(candidate)

    return {
        "path": str(dir_path),
        "rootPath": str(root.real_path),
        "parentPath": parent_path,
        "page": page_num,
        "pageSize": page_size,
        "total": total,
        "sort": sort_key,
        "order": sort_order,
        "includeHidden": include_hidden_b,
        "dirsOnly": dirs_only_b,
        "items": page_items,
    }


@router.post("/mkdir")
def fs_mkdir(payload: CreateDirRequest) -> dict:
    roots, _invalid = _get_configured_roots()
    parent_dir, root = _require_allowed_dir(payload.parent_path, roots)
    name = _validate_dir_name(payload.name)

    target = parent_dir / name
    try:
        target.relative_to(root.real_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid folder name.") from exc

    if target.exists():
        raise HTTPException(status_code=409, detail="An entry with that name already exists.")

    try:
        target.mkdir()
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="An entry with that name already exists.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied.") from exc
    except OSError as exc:
        message = exc.strerror.strip() if exc.strerror else "Unable to create folder."
        raise HTTPException(status_code=400, detail=message) from exc

    _cache_invalidate_dir(parent_dir)
    _cache_invalidate_dir(target)

    return {
        "ok": True,
        "path": str(target),
        "parentPath": str(parent_dir),
        "name": name,
    }
