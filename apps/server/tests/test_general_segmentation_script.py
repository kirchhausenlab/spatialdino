from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from spatialdino.segmentation.general import (
    apply_data_operations,
    apply_mask_operations,
    instance_segmentation,
    normalize_data_operations,
    normalize_mask_operations,
)


def _load_general_segmentation_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "post_processing" / "general_segmentation.py"
    spec = importlib.util.spec_from_file_location("general_segmentation_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load general segmentation module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


general_segmentation_script = _load_general_segmentation_module()


class GeneralSegmentationScriptTests(unittest.TestCase):
    def test_segments_raw_source_without_connected_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            output_root = general_segmentation_script.segment_general(
                input_dir,
                output_path=output_dir,
                source_kind="raw",
                source_folder="raw",
                threshold=5,
                component_index=0,
                invert_mask=False,
                run_ccl=False,
            )

            mask = tifffile.imread(output_root / "sample_a.tif")

        np.testing.assert_array_equal(mask, (raw >= 5).astype(np.uint8))

    def test_segments_pca_component_with_inverted_connected_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = input_dir / "raw"
            pca_dir = input_dir / "pca_2"
            raw_dir.mkdir(parents=True)
            pca_dir.mkdir()
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 3), dtype=np.uint8), photometric="minisblack")
            pca = np.zeros((2, 2, 3, 2), dtype=np.uint8)
            pca[..., 1] = np.array(
                [
                    [[0, 9, 9], [9, 9, 9]],
                    [[9, 9, 9], [9, 9, 1]],
                ],
                dtype=np.uint8,
            )
            tifffile.imwrite(pca_dir / "sample_a.tif", pca)

            output_root = general_segmentation_script.segment_general(
                input_dir,
                output_path=output_dir,
                source_kind="pca",
                source_folder="pca_2",
                threshold=5,
                component_index=1,
                invert_mask=True,
                run_ccl=True,
            )

            labels = tifffile.imread(output_root / "sample_a.tif")

        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(int(labels.max()), 2)

    def test_segments_feature_statistics_component_without_connected_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = input_dir / "raw"
            feature_stats_dir = input_dir / "feature_stats"
            raw_dir.mkdir(parents=True)
            feature_stats_dir.mkdir()
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 3), dtype=np.uint8), photometric="minisblack")
            feature_stats = np.zeros((2, 2, 3, 6), dtype=np.float32)
            feature_stats[..., 4] = np.array(
                [
                    [[0.0, 0.6, 0.2], [0.7, 0.1, 0.3]],
                    [[0.8, 0.4, 0.5], [0.2, 0.9, 0.0]],
                ],
                dtype=np.float32,
            )
            tifffile.imwrite(feature_stats_dir / "sample_a.tif", feature_stats)

            output_root = general_segmentation_script.segment_general(
                input_dir,
                output_path=output_dir,
                source_kind="feature_stats",
                source_folder="feature_stats",
                threshold=0.5,
                component_index=4,
                invert_mask=False,
                run_ccl=False,
            )

            mask = tifffile.imread(output_root / "sample_a.tif")

        np.testing.assert_array_equal(mask, (feature_stats[..., 4] >= 0.5).astype(np.uint8))

    def test_segments_with_data_and_mask_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.full((3, 3, 3), 9, dtype=np.uint8)
            raw[0, 0, 0] = 0
            raw[2, 2, 1] = 2
            raw[2, 2, 2] = 3
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            output_root = general_segmentation_script.segment_general(
                input_dir,
                output_path=output_dir,
                source_kind="raw",
                source_folder="raw",
                threshold=6,
                component_index=0,
                invert_mask=False,
                run_ccl=False,
                data_operations=[{"type": "invert_lut"}],
                mask_operations=[{"type": "remove_small_objects", "size": 2}],
                instance_method="none",
            )

            mask = tifffile.imread(output_root / "sample_a.tif")

        expected = (raw <= 3).astype(bool)
        expected[0, 0, 0] = False
        np.testing.assert_array_equal(mask, expected.astype(np.uint8))

    def test_new_data_operations_cpu_smoke(self) -> None:
        values = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
        operations = normalize_data_operations(
            [
                {
                    "type": "percentile_clipping",
                    "low_percentile": 1,
                    "high_percentile": 99,
                    "rescale": True,
                    "output_min": 0,
                    "output_max": 1,
                },
                {"type": "median_filter", "radius_z": 1, "radius_y": 1, "radius_x": 1},
                {
                    "type": "difference_of_gaussians",
                    "sigma_z": 0.5,
                    "sigma_y": 0.5,
                    "sigma_x": 0.5,
                    "sigma2_z": 1.0,
                    "sigma2_y": 1.0,
                    "sigma2_x": 1.0,
                    "response": "bright",
                },
                {"type": "top_hat", "radius": 1},
                {"type": "black_hat", "radius_z": 1, "radius_y": 1, "radius_x": 1},
            ]
        )

        processed = apply_data_operations(values, operations, source_kind="raw")

        self.assertEqual(processed.shape, values.shape)
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.isfinite(processed).all())

    def test_new_mask_operations(self) -> None:
        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[2:5, 2:5, 2:5] = True
        mask[3, 3, 3] = False
        mask[0, 0, 0] = True
        operations = normalize_mask_operations(
            [
                {"type": "fill_small_holes", "size": 1},
                {"type": "remove_border_objects"},
                {"type": "size_range", "min_size": 20, "max_size": 30},
            ]
        )

        processed = apply_mask_operations(mask, operations)

        self.assertTrue(processed[3, 3, 3])
        self.assertFalse(processed[0, 0, 0])
        self.assertEqual(int(processed.sum()), 27)

    def test_binary_mask_morphology_operations(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True

        dilated = apply_mask_operations(mask, normalize_mask_operations([{"type": "dilate", "radius": 1}]))
        closed = apply_mask_operations(dilated, normalize_mask_operations([{"type": "binary_closing", "radius": 1}]))
        opened = apply_mask_operations(mask, normalize_mask_operations([{"type": "binary_opening", "radius": 1}]))
        eroded = apply_mask_operations(dilated, normalize_mask_operations([{"type": "erode", "radius": 1}]))

        self.assertGreater(int(dilated.sum()), 1)
        self.assertTrue(closed[2, 2, 2])
        self.assertEqual(int(opened.sum()), 0)
        self.assertTrue(eroded[2, 2, 2])

    def test_distance_transform_watershed_splits_touching_objects(self) -> None:
        z_coords, y_coords, x_coords = np.ogrid[:7, :11, :11]
        first = (z_coords - 3) ** 2 + (y_coords - 5) ** 2 + (x_coords - 4) ** 2 <= 9
        second = (z_coords - 3) ** 2 + (y_coords - 5) ** 2 + (x_coords - 6) ** 2 <= 9
        mask = np.asarray(first | second, dtype=bool)

        connected = instance_segmentation(
            mask.astype(np.float32),
            mask,
            method="connected_components",
            voronoi_spot_sigma=2.0,
            voronoi_outline_sigma=2.0,
        )
        labels = instance_segmentation(
            mask.astype(np.float32),
            mask,
            method="distance_transform_watershed",
            voronoi_spot_sigma=2.0,
            voronoi_outline_sigma=2.0,
            distance_dynamic=0.0,
            distance_connectivity=6,
            distance_spacing_zyx=(1.0, 1.0, 1.0),
        )

        self.assertEqual(int(connected.max()), 1)
        self.assertGreaterEqual(int(labels.max()), 2)

    def test_intensity_prominence_watershed_splits_one_binary_component(self) -> None:
        z_coords, y_coords, x_coords = np.ogrid[:9, :21, :21]
        values = (
            np.exp(-(((z_coords - 4) ** 2 + (y_coords - 10) ** 2 + (x_coords - 8.5) ** 2) / (2 * 0.8**2)))
            + np.exp(-(((z_coords - 4) ** 2 + (y_coords - 10) ** 2 + (x_coords - 11.5) ** 2) / (2 * 0.8**2)))
        ).astype(np.float32)
        mask = values >= 0.005

        connected = instance_segmentation(
            values,
            mask,
            method="connected_components",
            voronoi_spot_sigma=2.0,
            voronoi_outline_sigma=2.0,
        )
        labels = instance_segmentation(
            values,
            mask,
            method="intensity_prominence_watershed",
            voronoi_spot_sigma=2.0,
            voronoi_outline_sigma=2.0,
            intensity_prominence=0.15,
            intensity_smoothing_sigma=0.0,
            intensity_low_percentile=1.0,
            intensity_high_percentile=99.0,
            intensity_connectivity=6,
        )

        self.assertEqual(int(connected.max()), 1)
        self.assertEqual(int(labels.max()), 2)

    def test_segment_general_with_distance_transform_watershed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            z_coords, y_coords, x_coords = np.ogrid[:7, :11, :11]
            first = (z_coords - 3) ** 2 + (y_coords - 5) ** 2 + (x_coords - 4) ** 2 <= 9
            second = (z_coords - 3) ** 2 + (y_coords - 5) ** 2 + (x_coords - 6) ** 2 <= 9
            raw = np.asarray(first | second, dtype=np.uint8)
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            output_root = general_segmentation_script.segment_general(
                input_dir,
                output_path=output_dir,
                source_kind="raw",
                source_folder="raw",
                threshold=1,
                component_index=0,
                invert_mask=False,
                run_ccl=False,
                instance_method="distance_transform_watershed",
                distance_transform_dynamic=0.0,
                distance_transform_connectivity=6,
                distance_transform_spacing=(1.0, 1.0, 1.0),
            )

            labels = tifffile.imread(output_root / "sample_a.tif")

        self.assertGreaterEqual(int(labels.max()), 2)

    def test_segment_general_with_intensity_prominence_watershed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            z_coords, y_coords, x_coords = np.ogrid[:9, :21, :21]
            raw = (
                np.exp(-(((z_coords - 4) ** 2 + (y_coords - 10) ** 2 + (x_coords - 8.5) ** 2) / (2 * 0.8**2)))
                + np.exp(-(((z_coords - 4) ** 2 + (y_coords - 10) ** 2 + (x_coords - 11.5) ** 2) / (2 * 0.8**2)))
            ).astype(np.float32)
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            output_root = general_segmentation_script.segment_general(
                input_dir,
                output_path=output_dir,
                source_kind="raw",
                source_folder="raw",
                threshold=0.005,
                component_index=0,
                invert_mask=False,
                run_ccl=False,
                instance_method="intensity_prominence_watershed",
                intensity_prominence=0.15,
                intensity_smoothing_sigma=0.0,
                intensity_percentiles=(1.0, 99.0),
                intensity_connectivity=6,
            )

            labels = tifffile.imread(output_root / "sample_a.tif")

        self.assertEqual(int(labels.max()), 2)


if __name__ == "__main__":
    unittest.main()
