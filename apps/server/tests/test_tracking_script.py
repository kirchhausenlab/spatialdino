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
        with patch.object(
            sys,
            "argv",
            [
                "tracking.py",
                "--input-path",
                "/tmp/tracking-input",
                "--segmentation-path",
                "/tmp/segmentations",
            ],
        ):
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

    def test_anisotropic_distance_scales_z_displacement(self) -> None:
        distance = tracking_script.anisotropic_distance(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 2.0]),
            zratio=3.0,
        )

        self.assertAlmostEqual(distance, 6.0)

    def test_run_tracking_writes_final_tracks_csv_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lr_feats").mkdir()
            (root / "raw").mkdir()
            segmentation_dir = root / "segmentations"
            segmentation_dir.mkdir()
            for index, x_start, amplitude in ((1, 1, 120), (2, 2, 140)):
                feats = np.zeros((2, 2, 2, 390), dtype=np.float32)
                np.save(root / "lr_feats" / f"stack{index:04d}.npy", feats)

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = 1
                tifffile.imwrite(segmentation_dir / f"stack{index:04d}.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(
                    root / "raw" / f"stack{index:04d}.tif",
                    raw,
                    photometric="minisblack",
                )

            output_path = tracking_script.run_tracking(
                root,
                segmentation_path=segmentation_dir,
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
            (root / "lr_feats").mkdir()
            (root / "raw").mkdir()
            segmentation_dir = root / "segmentations"
            segmentation_dir.mkdir()
            for index, x_start, amplitude in ((1, 1, 120), (2, 2, 140)):
                feats = np.zeros((2, 2, 2, 390), dtype=np.float32)
                np.save(root / "lr_feats" / f"stack{index:04d}.npy", feats)

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = 1
                tifffile.imwrite(segmentation_dir / f"stack{index:04d}.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(root / "raw" / f"stack{index:04d}.tif", raw, photometric="minisblack")

            output_path = tracking_script.run_tracking(
                root,
                segmentation_path=segmentation_dir,
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

    def test_run_tracking_can_write_to_custom_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lr_feats").mkdir()
            (root / "raw").mkdir()
            segmentation_dir = root / "segmentations"
            output_dir = root / "tracking-output"
            segmentation_dir.mkdir()
            for index, x_start, amplitude in ((1, 1, 120), (2, 2, 140)):
                feats = np.zeros((2, 2, 2, 390), dtype=np.float32)
                np.save(root / "lr_feats" / f"stack{index:04d}.npy", feats)

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = 1
                tifffile.imwrite(segmentation_dir / f"stack{index:04d}.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(root / "raw" / f"stack{index:04d}.tif", raw, photometric="minisblack")

            output_path = tracking_script.run_tracking(
                root,
                segmentation_path=segmentation_dir,
                output_path=output_dir,
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

        self.assertEqual(output_path, output_dir / "tracks.csv")

    def test_run_tracking_can_write_custom_filename_and_extended_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lr_feats").mkdir()
            (root / "raw").mkdir()
            segmentation_dir = root / "segmentations"
            segmentation_dir.mkdir()
            for index, label, x_start, amplitude in ((1, 7, 1, 120), (2, 9, 2, 140)):
                feats = np.zeros((2, 2, 2, 390), dtype=np.float32)
                np.save(root / "lr_feats" / f"stack{index:04d}.npy", feats)

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = label
                tifffile.imwrite(segmentation_dir / f"stack{index:04d}.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(root / "raw" / f"stack{index:04d}.tif", raw, photometric="minisblack")

            output_path = tracking_script.run_tracking(
                root,
                segmentation_path=segmentation_dir,
                output_filename="debug_tracks",
                params=tracking_script.TrackingParams(
                    max_distance_xy=20.0,
                    max_distance_z=10.0,
                    z_distance_weight=2.5,
                    min_distance_to_remove_cand=3.0,
                    vote_thresholds=(320, 300, 280, 260),
                    dice_threshold=0.5,
                    corr_threshold=0.5,
                    invert_z=False,
                    save_extended_results=True,
                ),
            )

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(output_path, root / "debug_tracks.csv")
        self.assertEqual(
            list(rows[0].keys()),
            [
                "track_id",
                "start",
                "t",
                "x",
                "y",
                "z",
                "A",
                "track_length",
                "stage",
                "assignment_method",
                "dice",
                "corr",
                "mse",
                "feat_votes",
                "vote_threshold",
                "anisotropic_distance",
                "label_id",
                "volume",
            ],
        )
        self.assertEqual(rows[0]["stage"], "NaN")
        self.assertEqual(rows[0]["label_id"], "7")
        self.assertEqual(rows[0]["volume"], "8")
        self.assertEqual(rows[1]["stage"], "1")
        self.assertEqual(rows[1]["assignment_method"], "distance_prefilter")
        self.assertAlmostEqual(float(rows[1]["dice"]), 1.0)
        self.assertAlmostEqual(float(rows[1]["anisotropic_distance"]), 1.0)
        self.assertEqual(rows[1]["label_id"], "9")

    def test_run_tracking_ignore_features_skips_feature_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lr_feats").mkdir()
            (root / "raw").mkdir()
            segmentation_dir = root / "segmentations"
            segmentation_dir.mkdir()
            for index, x_start, amplitude in ((1, 1, 120), (2, 2, 140)):
                np.save(root / "lr_feats" / f"stack{index:04d}.npy", np.zeros((1,), dtype=np.float32))

                seg = np.zeros((4, 4, 4), dtype=np.uint32)
                seg[1:3, 1:3, x_start : x_start + 2] = 1
                tifffile.imwrite(segmentation_dir / f"stack{index:04d}.tif", seg, photometric="minisblack")
                raw = np.zeros((4, 4, 4), dtype=np.uint16)
                raw[1:3, 1:3, x_start : x_start + 2] = amplitude
                tifffile.imwrite(root / "raw" / f"stack{index:04d}.tif", raw, photometric="minisblack")

            output_path = tracking_script.run_tracking(
                root,
                segmentation_path=segmentation_dir,
                params=tracking_script.TrackingParams(
                    max_distance_xy=20.0,
                    max_distance_z=10.0,
                    z_distance_weight=2.5,
                    min_distance_to_remove_cand=0.0,
                    vote_thresholds=(320, 300, 280, 260),
                    dice_threshold=0.5,
                    corr_threshold=0.5,
                    invert_z=False,
                    save_extended_results=True,
                    ignore_features=True,
                ),
            )

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[1]["stage"], "3")
        self.assertEqual(rows[1]["assignment_method"], "global_closest")
        self.assertEqual(rows[1]["corr"], "NaN")
        self.assertEqual(rows[1]["mse"], "NaN")
        self.assertEqual(rows[1]["feat_votes"], "NaN")

    def test_run_tracking_is_invariant_to_permuted_candidate_labels(self) -> None:
        def write_internal_volume(path: Path, volume_yxz: np.ndarray) -> None:
            tifffile.imwrite(path, np.moveaxis(volume_yxz, -1, 0), photometric="minisblack")

        def add_box(volume: np.ndarray, label: int, y: slice, x: slice, z: slice) -> None:
            volume[y, x, z] = label

        def centroid_links(output_path: Path) -> dict[tuple[float, float, float], tuple[float, float, float]]:
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            rows_by_track: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                rows_by_track.setdefault(row["track_id"], []).append(row)

            links: dict[tuple[float, float, float], tuple[float, float, float]] = {}
            for track_rows in rows_by_track.values():
                ordered = sorted(track_rows, key=lambda row: int(row["t"]))
                self.assertEqual(len(ordered), 2)
                start = tuple(round(float(ordered[0][axis]), 4) for axis in ("x", "y", "z"))
                stop = tuple(round(float(ordered[1][axis]), 4) for axis in ("x", "y", "z"))
                links[start] = stop
            return links

        def run_case(case_root: Path, *, frame_2_left_label: int, frame_2_right_label: int) -> dict[
            tuple[float, float, float], tuple[float, float, float]
        ]:
            (case_root / "lr_feats").mkdir(parents=True)
            (case_root / "raw").mkdir()
            segmentation_dir = case_root / "segmentations"
            segmentation_dir.mkdir()

            shape_yxz = (8, 7, 4)
            features = np.zeros((2, 2, 2, 390), dtype=np.float32)
            for index in (1, 2):
                np.save(case_root / "lr_feats" / f"stack{index:04d}.npy", features)

            seg_1 = np.zeros(shape_yxz, dtype=np.uint32)
            add_box(seg_1, 11, slice(2, 4), slice(3, 5), slice(1, 3))
            add_box(seg_1, 22, slice(6, 8), slice(3, 5), slice(1, 3))
            raw_1 = np.where(seg_1 == 11, 110, np.where(seg_1 == 22, 220, 0)).astype(np.uint16)

            seg_2 = np.zeros(shape_yxz, dtype=np.uint32)
            add_box(seg_2, frame_2_left_label, slice(4, 6), slice(1, 3), slice(1, 3))
            add_box(seg_2, frame_2_right_label, slice(4, 6), slice(5, 7), slice(1, 3))
            raw_2 = np.where(
                seg_2 == frame_2_left_label,
                130,
                np.where(seg_2 == frame_2_right_label, 240, 0),
            ).astype(np.uint16)

            write_internal_volume(segmentation_dir / "stack0001.tif", seg_1)
            write_internal_volume(segmentation_dir / "stack0002.tif", seg_2)
            write_internal_volume(case_root / "raw" / "stack0001.tif", raw_1)
            write_internal_volume(case_root / "raw" / "stack0002.tif", raw_2)

            output_path = tracking_script.run_tracking(
                case_root,
                segmentation_path=segmentation_dir,
                params=tracking_script.TrackingParams(
                    max_distance_xy=10.0,
                    max_distance_z=10.0,
                    z_distance_weight=1.0,
                    min_distance_to_remove_cand=0.0,
                    vote_thresholds=(320, 300, 280, 260),
                    dice_threshold=0.5,
                    corr_threshold=0.5,
                    invert_z=False,
                ),
            )
            return centroid_links(output_path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_links = run_case(root / "baseline", frame_2_left_label=101, frame_2_right_label=202)
            permuted_links = run_case(root / "permuted", frame_2_left_label=202, frame_2_right_label=101)

        self.assertEqual(permuted_links, baseline_links)


if __name__ == "__main__":
    unittest.main()
