from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


def _load_feature_mean_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "evaluation" / "feature_mean_tracking.py"
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("feature_mean_tracking", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature mean module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


feature_mean_tracking = _load_feature_mean_module()


def write_internal_volume(path: Path, volume_yxz: np.ndarray) -> None:
    tifffile.imwrite(path, np.moveaxis(volume_yxz, -1, 0), photometric="minisblack")


class FeatureMeanTrackingTests(unittest.TestCase):
    def test_three_frame_solver_uses_direct_consistency_term(self) -> None:
        cost01 = np.zeros((2, 2), dtype=np.float32)
        cost12 = np.zeros((2, 2), dtype=np.float32)
        cost02 = np.array(
            [
                [0.0, 10.0],
                [10.0, 0.0],
            ],
            dtype=np.float32,
        )

        triplets = feature_mean_tracking.solve_three_frame_assignment(
            cost01,
            cost12,
            cost02,
            direct_weight=1.0,
            candidate_top_k=2,
            time_limit_seconds=5.0,
        )

        self.assertEqual({(int(i), int(k)) for i, _j, k in triplets}, {(0, 0), (1, 1)})

    def test_feature_mean_tracks_objects_that_swap_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (6, 6, 4)
            for frame_index in (1, 2):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                if frame_index == 1:
                    segmentation[1:3, 1:3, 1:3] = 1
                    segmentation[3:5, 3:5, 1:3] = 2
                else:
                    segmentation[3:5, 3:5, 1:3] = 1
                    segmentation[1:3, 1:3, 1:3] = 2

                features_yxz = np.zeros((*shape_yxz, 3), dtype=np.float32)
                features_yxz[segmentation == 1] = np.array(
                    [1.0, 0.0, 0.0], dtype=np.float32
                )
                features_yxz[segmentation == 2] = np.array(
                    [0.0, 1.0, 0.0], dtype=np.float32
                )

                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = feature_mean_tracking.run_feature_mean_tracking(
                root,
                segmentation_dir,
                config=feature_mean_tracking.FeatureMeanConfig(
                    n_features=3,
                    samples_per_object=8,
                    device="cpu",
                    methods=(feature_mean_tracking.METHOD_FEATURE_MEAN,),
                    tracks_method=feature_mean_tracking.METHOD_FEATURE_MEAN,
                    max_frames=2,
                    progress=False,
                ),
            )

            output_path = root / "out"
            paths = feature_mean_tracking.save_result(result, output_path)
            tracks_path = Path(paths["tracks_csv"])
            with tracks_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                saved_fieldnames = reader.fieldnames
                saved_row_count = len(list(reader))

        self.assertEqual(result.frame_count, 2)
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.pair_metrics)
        assert result.metrics is not None
        self.assertEqual(float(result.metrics.iloc[0]["precision"]), 1.0)
        self.assertEqual(float(result.metrics.iloc[0]["recall"]), 1.0)
        self.assertEqual(float(result.metrics.iloc[0]["f1"]), 1.0)
        self.assertTrue(result.assignments["is_true_link"].all())
        self.assertEqual(set(result.assignments["method"]), {"feature_mean"})
        self.assertEqual(set(result.tracks["track_length"]), {2})
        self.assertEqual(set(result.tracks_by_method), {"feature_mean"})
        self.assertEqual(set(result.tracks["A"]), {1.0})
        self.assertEqual(
            list(result.tracks.columns), list(feature_mean_tracking.TRACK_COLUMNS)
        )
        self.assertEqual(saved_fieldnames, list(feature_mean_tracking.TRACK_COLUMNS))
        self.assertEqual(saved_row_count, 4)

    def test_default_search_window_blocks_far_feature_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (30, 6, 3)
            for frame_index in (1, 2):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                if frame_index == 1:
                    segmentation[1:3, 1:3, 1:2] = 1
                else:
                    segmentation[25:27, 1:3, 1:2] = 1

                features_yxz = np.zeros((*shape_yxz, 1), dtype=np.float32)
                features_yxz[segmentation == 1] = np.array([1.0], dtype=np.float32)
                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = feature_mean_tracking.run_feature_mean_tracking(
                root,
                segmentation_dir,
                config=feature_mean_tracking.FeatureMeanConfig(
                    n_features=1,
                    samples_per_object=4,
                    device="cpu",
                    methods=(feature_mean_tracking.METHOD_FEATURE_MEAN,),
                    tracks_method=feature_mean_tracking.METHOD_FEATURE_MEAN,
                    progress=False,
                ),
            )

        self.assertTrue(result.assignments.empty)
        self.assertEqual(len(result.tracks), 2)
        self.assertEqual(set(result.tracks["track_length"].astype(int)), {1})
        self.assertEqual(len(set(result.tracks["track_id"].astype(int))), 2)

    def test_three_frame_method_writes_conventional_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (7, 7, 4)
            positions = [
                {
                    1: (slice(1, 3), slice(1, 3), slice(1, 3)),
                    2: (slice(4, 6), slice(4, 6), slice(1, 3)),
                },
                {
                    1: (slice(2, 4), slice(1, 3), slice(1, 3)),
                    2: (slice(4, 6), slice(3, 5), slice(1, 3)),
                },
                {
                    1: (slice(3, 5), slice(1, 3), slice(1, 3)),
                    2: (slice(4, 6), slice(2, 4), slice(1, 3)),
                },
            ]
            for frame_index, frame_positions in enumerate(positions, start=1):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                for label_id, slices in frame_positions.items():
                    segmentation[slices] = label_id
                features_yxz = np.zeros((*shape_yxz, 2), dtype=np.float32)
                features_yxz[segmentation == 1] = np.array([1.0, 0.0], dtype=np.float32)
                features_yxz[segmentation == 2] = np.array([0.0, 1.0], dtype=np.float32)
                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = feature_mean_tracking.run_feature_mean_tracking(
                root,
                segmentation_dir,
                config=feature_mean_tracking.FeatureMeanConfig(
                    n_features=2,
                    samples_per_object=8,
                    device="cpu",
                    methods=(feature_mean_tracking.METHOD_CENTROID_FEATURE_3FRAME,),
                    tracks_method=feature_mean_tracking.METHOD_CENTROID_FEATURE_3FRAME,
                    three_frame_candidate_top_k=2,
                    three_frame_time_limit_seconds=5.0,
                    max_distance_xy=0.1,
                    max_distance_z=0.1,
                    progress=False,
                ),
            )
            paths = feature_mean_tracking.save_result(result, root / "out")
            with Path(paths["tracks_csv"]).open(newline="", encoding="utf-8") as handle:
                default_tracks_header = csv.DictReader(handle).fieldnames

        assert result.metrics is not None
        self.assertEqual(set(result.metrics["method"]), {"centroid_feature_3frame"})
        self.assertEqual(float(result.metrics.iloc[0]["f1"]), 1.0)
        self.assertEqual(set(result.tracks_by_method), {"centroid_feature_3frame"})
        self.assertEqual(
            default_tracks_header, list(feature_mean_tracking.TRACK_COLUMNS)
        )
        self.assertIn("tracks_centroid_feature_3frame_csv", paths)

    def test_prototype_method_writes_conventional_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (7, 7, 4)
            positions = [
                {
                    1: (slice(1, 3), slice(1, 3), slice(1, 3)),
                    2: (slice(4, 6), slice(4, 6), slice(1, 3)),
                },
                {
                    1: (slice(2, 4), slice(1, 3), slice(1, 3)),
                    2: (slice(4, 6), slice(3, 5), slice(1, 3)),
                },
                {
                    1: (slice(3, 5), slice(1, 3), slice(1, 3)),
                    2: (slice(4, 6), slice(2, 4), slice(1, 3)),
                },
            ]
            for frame_index, frame_positions in enumerate(positions, start=1):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                for label_id, slices in frame_positions.items():
                    segmentation[slices] = label_id
                features_yxz = np.zeros((*shape_yxz, 2), dtype=np.float32)
                features_yxz[segmentation == 1] = np.array([1.0, 0.0], dtype=np.float32)
                features_yxz[segmentation == 2] = np.array([0.0, 1.0], dtype=np.float32)
                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = feature_mean_tracking.run_feature_mean_tracking(
                root,
                segmentation_dir,
                config=feature_mean_tracking.FeatureMeanConfig(
                    n_features=2,
                    samples_per_object=8,
                    device="cpu",
                    methods=(feature_mean_tracking.METHOD_CENTROID_FEATURE_PROTOTYPE,),
                    tracks_method=(
                        feature_mean_tracking.METHOD_CENTROID_FEATURE_PROTOTYPE
                    ),
                    centroid_feature_weight=0.1,
                    progress=False,
                ),
            )
            paths = feature_mean_tracking.save_result(result, root / "out")
            with Path(paths["tracks_csv"]).open(newline="", encoding="utf-8") as handle:
                default_tracks_header = csv.DictReader(handle).fieldnames

        assert result.metrics is not None
        self.assertEqual(set(result.metrics["method"]), {"centroid_feature_prototype"})
        self.assertEqual(float(result.metrics.iloc[0]["f1"]), 1.0)
        self.assertEqual(set(result.tracks_by_method), {"centroid_feature_prototype"})
        self.assertEqual(
            set(result.assignments["prototype_observations"].astype(int)), {1, 2}
        )
        self.assertEqual(
            default_tracks_header, list(feature_mean_tracking.TRACK_COLUMNS)
        )
        self.assertIn("tracks_centroid_feature_prototype_csv", paths)

    def test_centroid_feature_keeps_centroid_solution_when_features_are_bad(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (8, 8, 4)
            for frame_index in (1, 2):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                if frame_index == 1:
                    segmentation[1:3, 1:3, 1:3] = 1
                    segmentation[5:7, 5:7, 1:3] = 2
                else:
                    segmentation[2:4, 1:3, 1:3] = 1
                    segmentation[5:7, 4:6, 1:3] = 2

                features_yxz = np.zeros((*shape_yxz, 2), dtype=np.float32)
                if frame_index == 1:
                    features_yxz[segmentation == 1] = np.array(
                        [1.0, 0.0], dtype=np.float32
                    )
                    features_yxz[segmentation == 2] = np.array(
                        [0.0, 1.0], dtype=np.float32
                    )
                else:
                    features_yxz[segmentation == 1] = np.array(
                        [0.0, 1.0], dtype=np.float32
                    )
                    features_yxz[segmentation == 2] = np.array(
                        [1.0, 0.0], dtype=np.float32
                    )

                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = feature_mean_tracking.run_feature_mean_tracking(
                root,
                segmentation_dir,
                config=feature_mean_tracking.FeatureMeanConfig(
                    n_features=2,
                    samples_per_object=8,
                    device="cpu",
                    methods=(
                        feature_mean_tracking.METHOD_CENTROID,
                        feature_mean_tracking.METHOD_FEATURE_MEAN,
                        feature_mean_tracking.METHOD_CENTROID_FEATURE,
                    ),
                    tracks_method=feature_mean_tracking.METHOD_CENTROID_FEATURE,
                    centroid_feature_weight=0.1,
                    progress=False,
                ),
            )
            paths = feature_mean_tracking.save_result(result, root / "out")
            with Path(paths["tracks_csv"]).open(newline="", encoding="utf-8") as handle:
                default_tracks_header = csv.DictReader(handle).fieldnames
            with Path(paths["tracks_centroid_feature_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                combined_tracks_header = csv.DictReader(handle).fieldnames

        assert result.metrics is not None
        metrics = result.metrics.set_index("method")
        self.assertEqual(float(metrics.loc["centroid", "f1"]), 1.0)
        self.assertEqual(float(metrics.loc["centroid_feature", "f1"]), 1.0)
        self.assertEqual(float(metrics.loc["feature_mean", "f1"]), 0.0)
        self.assertEqual(paths["tracks_default_method"], "centroid_feature")
        self.assertEqual(
            default_tracks_header, list(feature_mean_tracking.TRACK_COLUMNS)
        )
        self.assertEqual(
            combined_tracks_header, list(feature_mean_tracking.TRACK_COLUMNS)
        )

    def test_gt_metrics_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (4, 4, 3)
            for frame_index in (1, 2):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                segmentation[1:3, 1:3, 1:3] = 1
                features_yxz = np.zeros((*shape_yxz, 2), dtype=np.float32)
                features_yxz[segmentation == 1] = np.array([1.0, 0.0], dtype=np.float32)
                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = feature_mean_tracking.run_feature_mean_tracking(
                root,
                segmentation_dir,
                config=feature_mean_tracking.FeatureMeanConfig(
                    n_features=2,
                    samples_per_object=4,
                    device="cpu",
                    methods=(feature_mean_tracking.METHOD_CENTROID,),
                    tracks_method=feature_mean_tracking.METHOD_CENTROID,
                    compute_gt_metrics=False,
                    progress=False,
                ),
            )
            paths = feature_mean_tracking.save_result(result, root / "out")

        self.assertIsNone(result.metrics)
        self.assertIsNone(result.pair_metrics)
        self.assertNotIn("is_true_link", result.assignments.columns)
        self.assertNotIn("metrics_csv", paths)
        self.assertNotIn("pair_metrics_csv", paths)


if __name__ == "__main__":
    unittest.main()
