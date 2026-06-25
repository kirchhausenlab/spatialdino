from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "evaluation" / "eval_GT_tracks.py"
    spec = importlib.util.spec_from_file_location("eval_gt_tracks", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load eval module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_gt_tracks = _load_eval_module()


def write_internal_volume(path: Path, volume_yxz: np.ndarray) -> None:
    tifffile.imwrite(path, np.moveaxis(volume_yxz, -1, 0), photometric="minisblack")


def make_features(shape_yxz: tuple[int, int, int], *, offset: float) -> np.ndarray:
    y_size, x_size, z_size = shape_yxz
    feats = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                feats[z, y, x, 0] = y + offset
                feats[z, y, x, 1] = x + offset
                feats[z, y, x, 2] = z + offset
    return feats


def read_table(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [
            {key: "" if value is None else str(value) for key, value in row.items()}
            for row in pq.read_table(path).to_pylist()
        ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class EvalGTTracksTests(unittest.TestCase):
    def test_run_evaluation_writes_rows_arrays_and_candidate_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "inference"
            gt_path = root / "gt"
            output_path = root / "eval"
            (input_path / "lr_feats").mkdir(parents=True)
            (input_path / "raw").mkdir()
            gt_path.mkdir()

            shape = (5, 5, 5)
            for frame_index, x_shift in enumerate((0, 1), start=1):
                name = f"stack{frame_index:04d}"
                seg = np.zeros(shape, dtype=np.int64)
                seg[1:3, 1 + x_shift : 3 + x_shift, 1:3] = 1
                seg[3:5, 0:2, 1:3] = 2
                raw = np.zeros(shape, dtype=np.uint16)
                raw[seg == 1] = 100 + frame_index
                raw[seg == 2] = 200 + frame_index
                write_internal_volume(gt_path / f"{name}.tif", seg)
                write_internal_volume(input_path / "raw" / f"{name}.tif", raw)
                np.save(input_path / "lr_feats" / f"{name}.npy", make_features(shape, offset=float(frame_index)))

            manifest_path = eval_gt_tracks.run_evaluation(
                input_path,
                gt_segmentation_path=gt_path,
                output_path=output_path,
                params=eval_gt_tracks.EvaluationParams(
                    max_gap=1,
                    num_features="all",
                    max_distance_xy=5.0,
                    max_distance_z=5.0,
                    z_distance_weight=2.5,
                    no_compression=True,
                    overwrite=True,
                    candidate_batch_rows=2,
                ),
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_rows"], 4)
            self.assertEqual(manifest["array_shapes"]["feature_means"], [4, 3])
            feature_means = zarr.open(output_path / "feature_means.zarr", mode="r")
            gap_corr = zarr.open(output_path / "true_gap_corr.zarr", mode="r")
            gap_mse = zarr.open(output_path / "true_gap_mse.zarr", mode="r")
            self.assertEqual(feature_means.shape, (4, 3))
            self.assertEqual(gap_corr.shape, (4, 1, 3))
            self.assertEqual(gap_mse.shape, (4, 1, 3))
            self.assertTrue(np.isfinite(np.asarray(feature_means[0, :])).all())
            np.testing.assert_allclose(np.asarray(feature_means[0, :]), np.array([2.5, 2.5, 2.5]))

            rows_path = output_path / manifest["outputs"]["gt_rows"]
            rows = read_table(rows_path)
            first_label_row = next(row for row in rows if row["track_id"] == "1" and row["frame"] == "1")
            self.assertEqual(first_label_row["num_voxels"], "8")
            self.assertEqual(first_label_row["target_row_index_gap_1"], "2")
            self.assertAlmostEqual(float(first_label_row["dice_gap_1"]), 1.0)
            self.assertAlmostEqual(float(first_label_row["anisotropic_distance_gap_1"]), 1.0)

            candidate_path = output_path / manifest["outputs"]["candidate_pairs"]
            candidate_rows = read_table(candidate_path)
            true_rows = [
                row
                for row in candidate_rows
                if row["ref_track_id"] == "1" and row["cand_track_id"] == "1" and row["is_true_match"] == "True"
            ]
            self.assertTrue(true_rows)
            self.assertEqual(true_rows[0]["true_candidate_present"], "True")
            self.assertEqual(true_rows[0]["rank_distance"], "1")

            summary_path = output_path / manifest["outputs"]["candidate_ref_summary"]
            summary_rows = read_table(summary_path)
            label_summary = next(
                row for row in summary_rows if row["ref_track_id"] == "1" and row["ref_frame"] == "1"
            )
            self.assertEqual(label_summary["candidate_count"], "2")
            self.assertEqual(label_summary["true_candidate_present"], "True")


if __name__ == "__main__":
    unittest.main()
