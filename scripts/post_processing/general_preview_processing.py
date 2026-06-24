from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatialdino.segmentation.general import (
    DATA_BACKEND_GPU,
    apply_data_operations,
    normalize_data_operations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply General segmentation preview data operations.")
    parser.add_argument("--input-npy", required=True, help="Input NumPy volume.")
    parser.add_argument("--output-npy", required=True, help="Output NumPy volume.")
    parser.add_argument("--source-kind", required=True, help="General segmentation source kind.")
    parser.add_argument(
        "--data-operations-json",
        default="[]",
        help="JSON list of data operations to apply.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Visible GPU index. The parent process should narrow CUDA_VISIBLE_DEVICES first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operations = normalize_data_operations(json.loads(args.data_operations_json))
    values = np.asarray(np.load(Path(args.input_npy), mmap_mode="r"), dtype=np.float32)
    processed = apply_data_operations(
        values,
        operations,
        source_kind=str(args.source_kind),
        backend=DATA_BACKEND_GPU,
        gpu_index=int(args.gpu_index),
    )
    np.save(Path(args.output_npy), np.asarray(processed, dtype=np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
