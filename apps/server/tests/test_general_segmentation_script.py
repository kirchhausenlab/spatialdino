from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


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


if __name__ == "__main__":
    unittest.main()
