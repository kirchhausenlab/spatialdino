from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def _load_tracking_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "post_processing" / "tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tracking module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tracking_script = _load_tracking_module()


class TrackingScriptTests(unittest.TestCase):
    def test_parse_args_defaults_to_non_inverted_z_export(self) -> None:
        with patch.object(sys, "argv", ["tracking.py", "--input-path", "/tmp/tracking-input"]):
            args = tracking_script.parse_args()

        self.assertFalse(args.invert_z)

    def test_on_the_fly_feature_sampling_matches_trilinear_upsample(self) -> None:
        lr_feats = np.arange(2 * 2 * 2 * 1, dtype=np.float32).reshape(2, 2, 2, 1)
        chunk = tracking_script.load_feature_chunk_internal_yxz(lr_feats, start=0, end=1)
        axis_maps = tracking_script.build_axis_maps((2, 2, 2), (4, 4, 4))
        coords = np.array(
            [
                [0, 0, 0],
                [1, 2, 3],
                [3, 3, 3],
            ],
            dtype=np.int32,
        )

        sampled = tracking_script.sample_feature_chunk_at_internal_coords(chunk, axis_maps, coords)[0]

        hr_zyx = (
            F.interpolate(
                torch.from_numpy(np.moveaxis(lr_feats, -1, 0)).unsqueeze(0),
                size=(4, 4, 4),
                mode="trilinear",
                align_corners=False,
            )
            .cpu()
            .numpy()[0, 0]
        )
        hr_yxz = np.moveaxis(hr_zyx, 0, -1)
        expected = np.array([hr_yxz[tuple(coord)] for coord in coords], dtype=np.float32)

        np.testing.assert_allclose(sampled, expected, rtol=1e-5, atol=1e-5)

    def test_run_tracking_writes_final_tracks_csv_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, x_start, amplitude in ((1, 1, 120), (2, 2, 140)):
                sample_dir = root / f"stack{index:04d}"
                sample_dir.mkdir()

                feats = np.zeros((2, 2, 2, 390), dtype=np.float32)
                np.save(sample_dir / "lr_feats.npy", feats)

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = 1
                tifffile.imwrite(sample_dir / "instance_seg.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(
                    sample_dir / "volume_unnorm.tif",
                    raw,
                    photometric="minisblack",
                )

            output_path = tracking_script.run_tracking(
                root,
                segmentation_filename="instance_seg.tif",
                params=tracking_script.TrackingParams(
                    max_distance_xy=20.0,
                    max_distance_z=10.0,
                    z_distance_weight=2.5,
                    min_distance_to_remove_cand=3.0,
                    vote_thresholds=(320, 300, 280, 260),
                    dice_threshold=0.5,
                    corr_threshold=0.5,
                    invert_z=True,
                ),
            )

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            list(rows[0].keys()),
            ["track_id", "start", "t", "x", "y", "z", "A", "track_length"],
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["track_id"] for row in rows], ["1", "1"])
        self.assertEqual([row["start"] for row in rows], ["1", "1"])
        self.assertEqual([row["t"] for row in rows], ["1", "2"])
        self.assertEqual([row["track_length"] for row in rows], ["2", "2"])
        self.assertEqual([row["A"] for row in rows], ["120.0", "140.0"])
        self.assertAlmostEqual(float(rows[0]["x"]), 1.5)
        self.assertAlmostEqual(float(rows[1]["x"]), 2.5)
        self.assertAlmostEqual(float(rows[0]["y"]), 1.5)
        self.assertAlmostEqual(float(rows[0]["z"]), 2.5)

    def test_run_tracking_can_leave_z_unflipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, x_start, amplitude in ((1, 1, 120), (2, 2, 140)):
                sample_dir = root / f"stack{index:04d}"
                sample_dir.mkdir()

                feats = np.zeros((2, 2, 2, 390), dtype=np.float32)
                np.save(sample_dir / "lr_feats.npy", feats)

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = 1
                tifffile.imwrite(sample_dir / "instance_seg.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(sample_dir / "volume_unnorm.tif", raw, photometric="minisblack")

            output_path = tracking_script.run_tracking(
                root,
                segmentation_filename="instance_seg.tif",
                params=tracking_script.TrackingParams(
                    max_distance_xy=20.0,
                    max_distance_z=10.0,
                    z_distance_weight=2.5,
                    min_distance_to_remove_cand=3.0,
                    vote_thresholds=(320, 300, 280, 260),
                    dice_threshold=0.5,
                    corr_threshold=0.5,
                    invert_z=False,
                ),
            )

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertAlmostEqual(float(rows[0]["z"]), 1.5)


if __name__ == "__main__":
    unittest.main()
