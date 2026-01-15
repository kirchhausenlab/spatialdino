from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch

StorageKind = Literal["gpu", "cpu", "disk"]


def _numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    raise ValueError(
        "Disk storage only supports float16/float32. "
        f"Got dtype={dtype}."
    )


@dataclass
class TokenStore:
    tensor: torch.Tensor
    storage_kind: StorageKind
    num_special: int
    grid_size: Tuple[int, int, int]
    pin_memory: bool = False
    path: Optional[Path] = None
    memmap: Optional[np.memmap] = None
    _patch_view: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        shape: Tuple[int, int],
        dtype: torch.dtype,
        storage_kind: StorageKind,
        num_special: int,
        grid_size: Tuple[int, int, int],
        device: torch.device,
        pin_memory: bool = False,
        path: Optional[Path] = None,
    ) -> "TokenStore":
        if storage_kind == "gpu":
            tensor = torch.empty(shape, device=device, dtype=dtype)
            return cls(
                tensor=tensor,
                storage_kind=storage_kind,
                num_special=num_special,
                grid_size=grid_size,
                pin_memory=False,
            )

        if storage_kind == "cpu":
            tensor = torch.empty(
                shape,
                device="cpu",
                dtype=dtype,
                pin_memory=pin_memory,
            )
            return cls(
                tensor=tensor,
                storage_kind=storage_kind,
                num_special=num_special,
                grid_size=grid_size,
                pin_memory=pin_memory,
            )

        if storage_kind == "disk":
            if path is None:
                raise ValueError("Disk storage requires a path.")
            path.parent.mkdir(parents=True, exist_ok=True)
            np_dtype = _numpy_dtype_for_torch(dtype)
            memmap = np.memmap(
                path,
                mode="w+",
                dtype=np_dtype,
                shape=shape,
            )
            tensor = torch.from_numpy(memmap)
            return cls(
                tensor=tensor,
                storage_kind=storage_kind,
                num_special=num_special,
                grid_size=grid_size,
                pin_memory=False,
                path=path,
                memmap=memmap,
            )

        raise ValueError(f"Unknown storage_kind={storage_kind}")

    @property
    def patch_view(self) -> torch.Tensor:
        if self._patch_view is None:
            self._patch_view = self.tensor[self.num_special :].view(
                *self.grid_size, self.tensor.shape[1]
            )
        return self._patch_view

    def read(
        self,
        start: int,
        end: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        block = self.tensor[start:end]
        if dtype is not None and block.dtype != dtype:
            block = block.to(dtype=dtype)
        if self.storage_kind == "gpu":
            return block
        if device.type == "cpu":
            return block
        return block.to(device, non_blocking=self.pin_memory)

    def write(self, start: int, end: int, data: torch.Tensor) -> None:
        target = self.tensor[start:end]
        if target.device != data.device or target.dtype != data.dtype:
            data = data.to(device=target.device, dtype=target.dtype)
        target.copy_(data)

    def flush(self) -> None:
        if self.memmap is not None:
            self.memmap.flush()
