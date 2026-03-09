from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FsRoots:
    roots: list[Path]
    invalid: list[str]


def _configured_fs_roots_from_env() -> FsRoots:
    raw = (os.environ.get("SPATIALDINO_FS_ROOTS") or "").strip()
    if not raw:
        return FsRoots(roots=[Path(os.path.realpath("/"))], invalid=[])

    roots: list[Path] = []
    invalid: list[str] = []

    for part in raw.split(os.pathsep):
        text = part.strip()
        if not text:
            continue
        configured = Path(text).expanduser()
        if not configured.is_absolute():
            invalid.append(text)
            continue
        try:
            resolved = configured.resolve()
        except OSError:
            invalid.append(text)
            continue
        if not resolved.is_dir():
            invalid.append(text)
            continue
        roots.append(Path(os.path.realpath(resolved)))

    roots.sort(key=lambda path: (len(str(path)), str(path)), reverse=True)
    return FsRoots(roots=roots, invalid=invalid)
