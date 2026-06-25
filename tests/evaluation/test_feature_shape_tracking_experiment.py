from __future__ import annotations

import csv
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path

import numpy as np
import tifffile


def _load_experiment_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "scripts" / "evaluation" / "feature_shape_tracking_experiment.py"
    )
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "feature_shape_tracking_experiment", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment_module()


def write_internal_volume(path: Path, volume_yxz: np.ndarray) -> None:
    tifffile.imwrite(path, np.moveaxis(volume_yxz, -1, 0), photometric="minisblack")


class FeatureShapeTrackingExperimentTests(unittest.TestCase):
    def test_feature_methods_match_persistent_labels_when_objects_swap_positions(
        self,
    ) -> None:
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

            result = experiment.run_adjacent_tracking_experiment(
                root,
                segmentation_dir,
                config=experiment.ExperimentConfig(
                    n_features=3,
                    samples_per_object=8,
                    n_feature_projections=8,
                    n_shape_pairs=32,
                    n_shape_quantiles=8,
                    object_batch_size=2,
                    feature_channel_block=2,
                    mask_workers=1,
                    signature_workers=1,
                    devices=("cpu",),
                    pair_device="cpu",
                    torch_threads_per_worker=1,
                    progress=False,
                ),
            )

        summary = experiment.aggregate_summary(result.summary)
        self.assertEqual(
            set(summary["method"]),
            {"feature_mean", "sliced_wasserstein", "sliced_wasserstein_shape"},
        )
        self.assertTrue((summary["hungarian_accuracy"] == 1.0).all())
        self.assertTrue((summary["top1_accuracy"] == 1.0).all())
        self.assertEqual(int(summary["trackable_count"].min()), 2)
        self.assertTrue((summary["link_precision"] == 1.0).all())
        self.assertTrue((summary["link_recall"] == 1.0).all())
        self.assertTrue((summary["link_f1"] == 1.0).all())
        self.assertTrue((summary["true_link_outside_radius_count"] == 0).all())
        self.assertTrue((summary["assigned_outside_radius_count"] == 0).all())
        self.assertFalse(result.assignments.empty)
        self.assertFalse(result.link_metrics.empty)
        self.assertEqual(
            set(result.tracks),
            {"feature_mean", "sliced_wasserstein", "sliced_wasserstein_shape"},
        )
        self.assertFalse(result.track_points.empty)
        self.assertTrue(result.assignments["is_true_link"].all())
        for tracks in result.tracks.values():
            self.assertEqual(
                list(tracks.columns),
                ["track_id", "start", "t", "x", "y", "z", "A", "track_length"],
            )
            self.assertEqual(set(tracks["track_length"]), {2})

        with tempfile.TemporaryDirectory() as output_tmp:
            output_path = Path(output_tmp)
            paths = experiment.save_result(result, output_path)
            self.assertEqual(paths["tracks_default_method"], "feature_mean")

            for expected_path in (
                output_path / "tracks.csv",
                output_path / "tracks_feature_mean.csv",
                output_path / "tracks_by_method" / "feature_mean" / "tracks.csv",
            ):
                with expected_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, list(experiment.TRACK_COLUMNS))
                    self.assertGreater(len(list(reader)), 0)

    def test_max_frames_limits_discovered_timepoints_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (4, 4, 3)
            for frame_index in (1, 2, 3):
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

            result = experiment.run_adjacent_tracking_experiment(
                root,
                segmentation_dir,
                config=experiment.ExperimentConfig(
                    n_features=2,
                    samples_per_object=4,
                    n_feature_projections=4,
                    n_shape_pairs=8,
                    n_shape_quantiles=4,
                    object_batch_size=1,
                    feature_channel_block=2,
                    mask_workers=1,
                    signature_workers=1,
                    devices=("cpu",),
                    pair_device="cpu",
                    torch_threads_per_worker=1,
                    max_frames=2,
                    progress=False,
                ),
            )

        self.assertEqual(result.frame_count, 2)
        self.assertEqual(set(result.summary["ref_timepoint"]), {"stack0001"})
        self.assertEqual(set(result.summary["cand_timepoint"]), {"stack0002"})
        self.assertEqual(set(result.assignments["ref_timepoint"]), {"stack0001"})
        self.assertEqual(set(result.assignments["cand_timepoint"]), {"stack0002"})
        self.assertEqual(
            set(result.track_points["timepoint_name"]), {"stack0001", "stack0002"}
        )

    def test_tight_search_radius_can_gate_out_feature_correct_links(self) -> None:
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

            result = experiment.run_adjacent_tracking_experiment(
                root,
                segmentation_dir,
                config=experiment.ExperimentConfig(
                    n_features=2,
                    samples_per_object=8,
                    n_feature_projections=4,
                    n_shape_pairs=8,
                    n_shape_quantiles=4,
                    methods=(experiment.METHOD_FEATURE_MEAN,),
                    object_batch_size=2,
                    feature_channel_block=2,
                    mask_workers=1,
                    signature_workers=1,
                    devices=("cpu",),
                    pair_device="cpu",
                    torch_threads_per_worker=1,
                    search_radius_xy=0.75,
                    search_radius_z=0.75,
                    progress=False,
                ),
            )

        summary = experiment.aggregate_summary(result.summary)
        self.assertEqual(float(summary.iloc[0]["link_f1"]), 0.0)
        self.assertEqual(int(summary.iloc[0]["true_link_outside_radius_count"]), 2)
        self.assertEqual(int(summary.iloc[0]["assigned_outside_radius_count"]), 0)
        self.assertTrue(result.assignments["inside_search_radius"].all())
        self.assertFalse(result.assignments["is_true_link"].any())

    def test_search_radius_relaxes_refs_with_no_allowed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "lr_feats"
            segmentation_dir = root / "gt"
            feature_dir.mkdir()
            segmentation_dir.mkdir()

            shape_yxz = (8, 8, 4)
            positions = {
                1: (slice(1, 3), slice(1, 3), slice(1, 3)),
                2: (slice(5, 7), slice(5, 7), slice(1, 3)),
            }
            for frame_index in (1, 2):
                segmentation = np.zeros(shape_yxz, dtype=np.uint16)
                segmentation[positions[frame_index]] = 1
                features_yxz = np.zeros((*shape_yxz, 2), dtype=np.float32)
                features_yxz[segmentation == 1] = np.array([1.0, 0.0], dtype=np.float32)

                np.save(
                    feature_dir / f"stack{frame_index:04d}.npy",
                    np.transpose(features_yxz, (2, 0, 1, 3)),
                )
                write_internal_volume(
                    segmentation_dir / f"stack{frame_index:04d}.tif", segmentation
                )

            result = experiment.run_adjacent_tracking_experiment(
                root,
                segmentation_dir,
                config=experiment.ExperimentConfig(
                    n_features=2,
                    samples_per_object=4,
                    n_feature_projections=4,
                    n_shape_pairs=8,
                    n_shape_quantiles=4,
                    methods=(experiment.METHOD_FEATURE_MEAN,),
                    object_batch_size=1,
                    feature_channel_block=2,
                    mask_workers=1,
                    signature_workers=1,
                    devices=("cpu",),
                    pair_device="cpu",
                    torch_threads_per_worker=1,
                    search_radius_xy=0.5,
                    search_radius_z=0.5,
                    progress=False,
                ),
            )

        summary = experiment.aggregate_summary(result.summary)
        self.assertEqual(float(summary.iloc[0]["link_f1"]), 1.0)
        self.assertEqual(int(summary.iloc[0]["radius_relaxed_ref_count"]), 1)
        self.assertEqual(int(summary.iloc[0]["true_link_outside_radius_count"]), 1)
        self.assertEqual(int(summary.iloc[0]["assigned_outside_radius_count"]), 1)
        self.assertTrue(result.assignments["radius_relaxed_for_ref"].all())
        self.assertFalse(result.assignments["inside_search_radius"].any())


if __name__ == "__main__":
    unittest.main()
