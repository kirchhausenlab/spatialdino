from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


def _load_probmap_segmentation_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "post_processing" / "probmap_segmentation.py"
    spec = importlib.util.spec_from_file_location("probmap_segmentation_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probability-map segmentation module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probmap_segmentation_script = _load_probmap_segmentation_module()


class ProbmapSegmentationScriptTests(unittest.TestCase):
    def test_lists_probability_map_timepoints_without_lr_feats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            probmap_dir = root / "probmap"
            raw_dir.mkdir()
            probmap_dir.mkdir()
            for name in ("sample_2", "sample_10"):
                tifffile.imwrite(raw_dir / f"{name}.tif", np.zeros((2, 2, 2), dtype=np.uint8), photometric="minisblack")
                tifffile.imwrite(
                    probmap_dir / f"{name}.tif",
                    np.zeros((2, 2, 2), dtype=np.float32),
                    photometric="minisblack",
                )

            timepoints = probmap_segmentation_script.list_probability_map_timepoints(root)

        self.assertEqual([name for name, _path in timepoints], ["sample_2", "sample_10"])

    def test_segments_probability_maps_without_lr_feats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            raw_dir = root / "raw"
            probmap_dir = root / "probmap"
            raw_dir.mkdir()
            probmap_dir.mkdir()
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 2), dtype=np.uint8), photometric="minisblack")
            probmap = np.array(
                [
                    [[0.1, 0.7], [0.8, 0.2]],
                    [[0.9, 0.4], [0.3, 0.6]],
                ],
                dtype=np.float32,
            )
            tifffile.imwrite(probmap_dir / "sample_a.tif", probmap, photometric="minisblack")

            output_root = probmap_segmentation_script.segment_probability_maps(
                root,
                output_path=output_dir,
                threshold=0.5,
                run_ccl=False,
            )

            mask = tifffile.imread(output_root / "sample_a.tif")

        np.testing.assert_array_equal(mask, (probmap >= 0.5).astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
