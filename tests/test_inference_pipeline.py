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
    patch_size: tuple[int, int, int] = (2, 2, 2),
    stride: tuple[int, int, int] = (2, 2, 2),
    padding_mode: str = "reflect",
    global_hist_min: float | None = None,
    global_hist_max: float | None = None,
):
    return OmegaConf.create(
        {
            "file_path": str(input_path),
            "save_path": str(output_path),
            "upsample_factor": float(upsample_factor),
            "isotropic_scale_factor": list(isotropic_scale_factor),
            "patch_size": list(patch_size),
            "stride": list(stride),
            "crop_params": list(crop_params),
            "chunk_size": list(chunk_size),
            "padding_mode": padding_mode,
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

    def test_save_feature_metadata_writes_json_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feats_path = Path(tmp) / "sample.npy"

            inference_script.save_feature_metadata(
                feats_path,
                {
                    "padding_mode": "reflect",
                    "pad_before": (3, 0, 0),
                    "pad_after": (3, 0, 0),
                },
                model_input_shape=(216, 12, 12),
                lr_feats_shape=(27, 6, 6, 390),
            )

            metadata_path = feats_path.with_name("sample_metadata.json")
            self.assertTrue(metadata_path.exists())
            self.assertIn('"padding_mode": "reflect"', metadata_path.read_text())
            self.assertIn('"model_input_shape": [', metadata_path.read_text())

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
                item = dataset[0]

            expected = volume[1:3, 1:4, 2:5]
            np.testing.assert_array_equal(imsave.call_args.args[1], expected)
            saved_path = Path(imsave.call_args.args[0])
            self.assertEqual(saved_path.parts[-2:], ("raw", "sample.tif"))
            self.assertEqual(item["vol_metadata"]["timepoint_name"], "sample")
            self.assertEqual(Path(item["vol_metadata"]["raw_path"]).parts[-2:], ("raw", "sample.tif"))
            self.assertEqual(Path(item["vol_metadata"]["lr_feats_path"]).parts[-2:], ("lr_feats", "sample.npy"))

    def test_inference_dataset_uses_minimal_patch_padding_not_chunk_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.arange(5 * 7 * 9, dtype=np.uint16).reshape(5, 7, 9)
            input_path = root / "sample.tiff"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                chunk_size=(8, 8, 8),
                global_hist_min=0.0,
                global_hist_max=500.0,
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])

            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ):
                item = dataset[0]

            self.assertEqual(item["vol_metadata"]["pre_pad_shape"], (5, 7, 9))
            self.assertEqual(item["vol_metadata"]["target_vol_size"], (6, 8, 10))
            self.assertEqual(item["vol_metadata"]["padding"], (1, 1, 1))
            self.assertEqual(item["vol_metadata"]["pad_before"], (0, 0, 0))
            self.assertEqual(item["vol_metadata"]["pad_after"], (1, 1, 1))

    def test_inference_dataset_records_reflect_padding_for_70_plane_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.full((70, 4, 4), 100, dtype=np.uint16)
            input_path = root / "sample.tiff"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                upsample_factor=3.0,
                patch_size=(8, 2, 2),
                stride=(8, 2, 2),
                chunk_size=(8, 2, 2),
                global_hist_min=0.0,
                global_hist_max=200.0,
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])

            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ):
                item = dataset[0]

            self.assertEqual(item["vol_metadata"]["pre_pad_shape"], (210, 12, 12))
            self.assertEqual(item["vol_metadata"]["target_vol_size"], (216, 12, 12))
            self.assertEqual(item["vol_metadata"]["padding"], (6, 0, 0))
            self.assertEqual(item["vol_metadata"]["pad_before"], (3, 0, 0))
            self.assertEqual(item["vol_metadata"]["pad_after"], (3, 0, 0))
            self.assertEqual(item["vol_metadata"]["effective_padding_mode"], "reflect")

    def test_inference_dataset_sanitizes_non_finite_voxels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.array(
                [
                    [[1.0, np.nan], [2.0, np.inf]],
                    [[3.0, -np.inf], [4.0, 5.0]],
                ],
                dtype=np.float32,
            )
            input_path = root / "sample.tiff"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                chunk_size=(2, 2, 2),
                global_hist_min=0.0,
                global_hist_max=5.0,
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])

            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ):
                item = dataset[0]

            self.assertTrue(np.isfinite(item["image"]).all())
            self.assertFalse(np.isnan(item["image"]).any())
            self.assertFalse(np.isinf(item["image"]).any())

    def test_inference_dataset_rejects_volume_with_no_finite_voxels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.full((2, 2, 2), np.nan, dtype=np.float32)
            input_path = root / "invalid.tiff"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                chunk_size=(2, 2, 2),
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])

            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ):
                with self.assertRaisesRegex(ValueError, "contains no finite voxels"):
                    _ = dataset[0]

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

    def test_inference_transform_does_not_delete_planes_when_mask_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            image = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
            config = _make_config(
                input_path=root,
                output_path=output_dir,
                chunk_size=(2, 4, 6),
                global_hist_min=0.0,
                global_hist_max=1.0,
            )
            transform = InferenceTransform(config=config)
            res = transform(
                data={"image": image, "mask": np.zeros(image.shape[0], dtype=bool)},
                vol_metadata={
                    "target_vol_size": image.shape,
                    "save_path": str(output_dir / "sample"),
                },
                chunk_interpolate=False,
                device="cpu",
            )

            expected = image[None]
            np.testing.assert_allclose(res["volume"].numpy(), expected)

    def test_inference_transform_reflect_pads_volume_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            image = np.stack(
                [
                    np.full((2, 2), 1, dtype=np.float32),
                    np.full((2, 2), 2, dtype=np.float32),
                ],
                axis=0,
            )
            config = _make_config(
                input_path=root,
                output_path=output_dir,
                chunk_size=(2, 2, 2),
                global_hist_min=0.0,
                global_hist_max=2.0,
            )
            transform = InferenceTransform(config=config)
            res = transform(
                data={"image": image, "mask": np.zeros(image.shape[0], dtype=bool)},
                vol_metadata={
                    "target_vol_size": (4, 2, 2),
                    "effective_padding_mode": "reflect",
                    "save_path": str(output_dir / "sample"),
                },
                chunk_interpolate=False,
                device="cpu",
            )

            expected_z = np.array([2, 1, 2, 1], dtype=np.float32)
            np.testing.assert_allclose(res["volume"].numpy()[0, :, 0, 0], expected_z)

    def test_inference_transform_falls_back_to_replicate_for_tiny_reflect_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()

            volume = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
            input_path = root / "sample.tiff"

            config = _make_config(
                input_path=root,
                output_path=output_dir,
                patch_size=(2, 2, 2),
                stride=(2, 2, 2),
                chunk_size=(2, 2, 2),
                global_hist_min=0.0,
                global_hist_max=4.0,
            )
            dataset = InferenceDataset(config=config, fnames=[input_path])
            with patch("spatialdino.data.inference.io.imread", return_value=volume), patch(
                "spatialdino.data.inference.io.imsave"
            ):
                item = dataset[0]

            self.assertEqual(item["vol_metadata"]["effective_padding_mode"], "replicate")

            transform = InferenceTransform(config=config)
            res = transform(
                data={"image": item["image"], "mask": item["mask"]},
                vol_metadata=item["vol_metadata"],
                chunk_interpolate=False,
                device="cpu",
            )

            self.assertEqual(tuple(res["volume"].shape), (1, 2, 2, 2))
            np.testing.assert_allclose(
                res["volume"].numpy()[0, 0],
                res["volume"].numpy()[0, 1],
            )
