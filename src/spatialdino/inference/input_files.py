from __future__ import annotations

import os
from pathlib import Path
from natsort import natsorted


def list_tiff_paths(input_path: str | Path) -> list[Path]:
    root = Path(input_path).expanduser()
    tiff_paths: list[Path] = []

    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith((".tif", ".tiff")):
                continue
            tiff_paths.append(Path(entry.path))

    return natsorted(tiff_paths, key=lambda path: path.name)
