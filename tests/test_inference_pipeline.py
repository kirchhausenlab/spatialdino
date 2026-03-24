from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from spatialdino.data.inference import InferenceDataset, InferenceTransform
from spatialdino.inference.input_files import list_tiff_paths


def _load_inference_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "inference" / "inference.py"
    spec = importlib.util.spec_from_file_location("inference_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load inference module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inference_script = _load_inference_module()


def _make_config(
    *,
    input_path: Path,
    output_path: Path,
    upsample_factor: float = 1.0,
    isotropic_scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    crop_params: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    chunk_size: tuple[int, int, int] = (4, 4, 4),
    global_hist_min: float | None = None,
    global_hist_max: float | None = None,
):
    return OmegaConf.create(
        {
            "file_path": str(input_path),
            "save_path": str(output_path),
            "upsample_factor": float(upsample_factor),
            "isotropic_scale_factor": list(isotropic_scale_factor),
            "patch_size": [2, 2, 2],
            "stride": [2, 2, 2],
            "crop_params": list(crop_params),
            "chunk_size": list(chunk_size),
            "dtype": "fp32",
            "global_hist_min": global_hist_min,
            "global_hist_max": global_hist_max,
            "in_chans": 1,
            "mean": [0.0],
            "std": [1.0],
        }
    )


class InferencePipelineTests(unittest.TestCase):
    def test_list_tiff_paths_matches_webgui_file_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.tif").write_bytes(b"")
            (root / "A.tiff").write_bytes(b"")
            (root / ".hidden.tiff").write_bytes(b"")
            (root / "ignore.txt").write_bytes(b"")
            (root / "subdir").mkdir()

            paths = list_tiff_paths(root)

            self.assertEqual([path.name for path in paths], ["A.tiff", "b.tif"])

    def test_iter_batch_samples_yields_every_batch_item(self) -> None:
        batch = {
            "images": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "masks": np.array([[True, False], [False, True]]),
            "vol_metadata": [
                {"save_path": "sample_a"},
                {"save_path": "sample_b"},
            ],
        }

        samples = list(inference_script.iter_batch_samples(batch))

        self.assertEqual(len(samples), 2)
        np.testing.assert_array_equal(samples[0][0]["image"], batch["images"][0])
        np.testing.assert_array_equal(samples[1][0]["mask"], batch["masks"][1])
        self.assertEqual(samples[0][1]["save_path"], "sample_a")
        self.assertEqual(samples[1][1]["save_path"], "sample_b")

    def test_inference_dataset_saves_cropped_volume_unnorm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
            input_path = root / "sample.tiff"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                crop_params=(1, 3, 1, 4, 2, 5),
                chunk_size=(2, 4, 4),
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])

            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ) as imsave:
                _ = dataset[0]

            expected = volume[1:3, 1:4, 2:5]
            np.testing.assert_array_equal(imsave.call_args.args[1], expected)
            self.assertEqual(Path(imsave.call_args.args[0]).name, "volume_unnorm.tif")

    def test_inference_transform_does_not_apply_anisotropy_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.stack(
                [
                    np.full((4, 6), 10, dtype=np.uint16),
                    np.full((4, 6), 90, dtype=np.uint16),
                ],
                axis=0,
            )
            input_path = root / "anisotropic.tif"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                upsample_factor=1.0,
                isotropic_scale_factor=(2.0, 1.0, 1.0),
                chunk_size=(4, 4, 6),
                global_hist_min=0.0,
                global_hist_max=100.0,
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])
            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ):
                item = dataset[0]

            self.assertEqual(item["image"].shape, (4, 4, 6))
            self.assertEqual(item["vol_metadata"]["target_vol_size"], (4, 4, 6))
            self.assertEqual(item["vol_metadata"]["padding"], (0, 0, 0))

            transform = InferenceTransform(config=config)
            res = transform(
                data={"image": item["image"], "mask": item["mask"]},
                vol_metadata=item["vol_metadata"],
                chunk_interpolate=False,
                device="cpu",
            )

            np.testing.assert_allclose(
                res["volume"].squeeze(0).numpy(),
                item["image"],
                rtol=1e-5,
                atol=1e-5,
            )
