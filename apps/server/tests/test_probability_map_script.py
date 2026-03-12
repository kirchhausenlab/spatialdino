from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def _load_probability_map_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "post_processing" / "probability_map.py"
    spec = importlib.util.spec_from_file_location("probability_map_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probability-map module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probability_map_script = _load_probability_map_module()


def _collect_samples_reference(
    timepoint: object,
    *,
    seg_tif: Path,
    valid_mask_tif: Path | None,
    max_samples_per_class: int,
    feature_batch: int,
    seed: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    lr_feats = np.load(timepoint.lr_path, mmap_mode="r")
    raw_shape = probability_map_script.read_tiff_shape(timepoint.raw_path)

    seg = probability_map_script.read_tiff_volume(seg_tif)
    valid_mask = (
        probability_map_script.load_valid_mask(valid_mask_tif, shape_zyx=raw_shape)
        if valid_mask_tif is not None
        else np.ones(raw_shape, dtype=bool)
    )
    bg_label = int(np.min(seg))
    bg_mask = (seg == bg_label) & valid_mask
    fg_mask = (seg != bg_label) & valid_mask
    bg_idx_per_z, fg_idx_per_z = probability_map_script.precompute_mask_indices(bg_mask, fg_mask)

    sampler = probability_map_script.UpsampledFeatureSampler(lr_feats, raw_shape, device)
    mins, maxs = probability_map_script.compute_feature_min_max_from_lr(lr_feats, feature_batch=feature_batch)
    feature_names = probability_map_script.build_feature_names(sampler.channel_count)
    bg_list: list[np.ndarray] = [np.empty((0,), dtype=np.float32) for _ in range(sampler.channel_count)]
    fg_list: list[np.ndarray] = [np.empty((0,), dtype=np.float32) for _ in range(sampler.channel_count)]
    per_z_budget = int(np.ceil(max_samples_per_class / max(1, raw_shape[0])))
    rngs = [np.random.default_rng(seed + feature_index) for feature_index in range(sampler.channel_count)]
    z_chunks = probability_map_script.iter_z_index_chunks(raw_shape[0])

    for start in range(0, sampler.channel_count, feature_batch):
        end = min(sampler.channel_count, start + feature_batch)
        bg_parts = [[] for _ in range(end - start)]
        fg_parts = [[] for _ in range(end - start)]
        batch_mins_t = torch.from_numpy(mins[start:end]).to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        batch_maxs_t = torch.from_numpy(maxs[start:end]).to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)

        for z_indices in z_chunks:
            sampled = sampler.sample_feature_range(start, end, z_indices)
            quantized = probability_map_script.quantize_feature_batch(sampled, batch_mins_t, batch_maxs_t)
            quantized_np = quantized.reshape(end - start, len(z_indices), -1).detach().cpu().numpy()
            for local_index, z_index in enumerate(z_indices):
                bg_indices = bg_idx_per_z[z_index]
                fg_indices = fg_idx_per_z[z_index]
                for offset, feature_index in enumerate(range(start, end)):
                    feature_rng = rngs[feature_index]
                    if bg_indices.size > 0:
                        bg_count = min(per_z_budget, int(bg_indices.size))
                        picked_bg = feature_rng.choice(bg_indices, size=bg_count, replace=False)
                        bg_parts[offset].append(quantized_np[offset, local_index, picked_bg])
                    if fg_indices.size > 0:
                        fg_count = min(per_z_budget, int(fg_indices.size))
                        picked_fg = feature_rng.choice(fg_indices, size=fg_count, replace=False)
                        fg_parts[offset].append(quantized_np[offset, local_index, picked_fg])

        for offset, feature_index in enumerate(range(start, end)):
            feature_rng = rngs[feature_index]
            bg_values = (
                np.concatenate(bg_parts[offset]).astype(np.float32, copy=False)
                if bg_parts[offset]
                else np.empty((0,), dtype=np.float32)
            )
            fg_values = (
                np.concatenate(fg_parts[offset]).astype(np.float32, copy=False)
                if fg_parts[offset]
                else np.empty((0,), dtype=np.float32)
            )
            if bg_values.size > max_samples_per_class:
                bg_values = bg_values[feature_rng.permutation(bg_values.size)[:max_samples_per_class]]
            if fg_values.size > max_samples_per_class:
                fg_values = fg_values[feature_rng.permutation(fg_values.size)[:max_samples_per_class]]
            bg_list[feature_index] = bg_values
            fg_list[feature_index] = fg_values

    return bg_list, fg_list, feature_names


class ProbabilityMapScriptTests(unittest.TestCase):
    def test_on_the_fly_sampling_matches_trilinear_interpolate(self) -> None:
        lr_feats = np.arange(2 * 3 * 4 * 1, dtype=np.float32).reshape(2, 3, 4, 1)
        sampler = probability_map_script.UpsampledFeatureSampler(
            lr_feats,
            target_shape=(4, 6, 8),
            device=torch.device("cpu"),
        )
        sampled = sampler.sample_feature_range(0, 1, [0, 2, 3]).cpu().numpy()[0]

        expected = (
            F.interpolate(
                torch.from_numpy(np.moveaxis(lr_feats[..., :1], -1, 0)).unsqueeze(0),
                size=(4, 6, 8),
                mode="trilinear",
                align_corners=False,
            )
            .cpu()
            .numpy()[0, 0, [0, 2, 3]]
        )
        np.testing.assert_allclose(sampled, expected, rtol=1e-5, atol=1e-5)

    def test_feature_min_max_from_lr_reads_per_feature_ranges(self) -> None:
        lr_feats = np.array(
            [
                [[[0.0, 10.0], [2.0, 8.0]], [[4.0, 6.0], [6.0, 4.0]]],
                [[[8.0, 2.0], [10.0, 0.0]], [[12.0, -2.0], [14.0, -4.0]]],
            ],
            dtype=np.float32,
        )

        mins, maxs = probability_map_script.compute_feature_min_max_from_lr(lr_feats, feature_batch=1)

        np.testing.assert_allclose(mins, np.array([0.0, -4.0], dtype=np.float32), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(maxs, np.array([14.0, 10.0], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_semantic_to_instance_seg_labels_connected_components(self) -> None:
        semantic_seg = np.zeros((3, 3, 3), dtype=np.uint8)
        semantic_seg[0, 0, 0] = 1
        semantic_seg[2, 2, 2] = 1

        instance_seg = probability_map_script.semantic_to_instance_seg(semantic_seg)

        self.assertEqual(instance_seg.dtype, np.uint32)
        self.assertEqual(int(instance_seg[0, 0, 0]), 1)
        self.assertEqual(int(instance_seg[2, 2, 2]), 2)
        self.assertEqual(int(instance_seg.max()), 2)

    def test_collect_samples_matches_reference_quantized_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample_a"
            sample_dir.mkdir()
            seg_tif = root / "seg.tif"

            lr_feats = np.array(
                [
                    [[[0.0, 10.0, 2.0], [1.0, 9.0, 3.0]], [[2.0, 8.0, 4.0], [3.0, 7.0, 5.0]]],
                    [[[4.0, 6.0, 6.0], [5.0, 5.0, 7.0]], [[6.0, 4.0, 8.0], [7.0, 3.0, 9.0]]],
                ],
                dtype=np.float32,
            )
            np.save(sample_dir / "lr_feats.npy", lr_feats)
            tifffile.imwrite(sample_dir / "volume_unnorm.tif", np.zeros((4, 4, 4), dtype=np.uint16))
            seg = np.zeros((4, 4, 4), dtype=np.uint8)
            seg[:, :, 2:] = 1
            tifffile.imwrite(seg_tif, seg)

            timepoint = probability_map_script.TimepointPaths(
                name="sample_a",
                subfolder=sample_dir,
                lr_path=sample_dir / "lr_feats.npy",
                raw_path=sample_dir / "volume_unnorm.tif",
            )

            actual_bg, actual_fg, actual_names = probability_map_script.collect_samples_from_timepoint(
                timepoint,
                seg_tif=seg_tif,
                valid_mask_tif=None,
                max_samples_per_class=12,
                feature_batch=2,
                seed=1337,
                device=torch.device("cpu"),
            )
            expected_bg, expected_fg, expected_names = _collect_samples_reference(
                timepoint,
                seg_tif=seg_tif,
                valid_mask_tif=None,
                max_samples_per_class=12,
                feature_batch=2,
                seed=1337,
                device=torch.device("cpu"),
            )

            self.assertEqual(actual_names, expected_names)
            for actual_values, expected_values in zip(actual_bg, expected_bg, strict=True):
                np.testing.assert_array_equal(actual_values, expected_values)
            for actual_values, expected_values in zip(actual_fg, expected_fg, strict=True):
                np.testing.assert_array_equal(actual_values, expected_values)

    def test_run_probability_map_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seg_tif = root / "seg.tif"
            seg = np.zeros((4, 4, 4), dtype=np.uint8)
            seg[:, :, 2:] = 1
            tifffile.imwrite(seg_tif, seg)

            base_feature0 = np.zeros((2, 2, 2), dtype=np.float32)
            base_feature0[:, :, 1] = 10.0
            base_feature1 = np.zeros((2, 2, 2), dtype=np.float32)
            base_feature1[:, 1, :] = 5.0

            for offset, name in enumerate(("sample_a", "sample_b")):
                sample_dir = root / name
                sample_dir.mkdir()
                feats = np.stack((base_feature0 + offset, base_feature1 + offset), axis=-1)
                np.save(sample_dir / "lr_feats.npy", feats)
                tifffile.imwrite(sample_dir / "volume_unnorm.tif", np.zeros((4, 4, 4), dtype=np.uint16))

            params = probability_map_script.ProbabilityMapParams(
                run_density_estimation=True,
                training_timepoint="sample_a",
                seg_tif=seg_tif,
                valid_mask_tif=None,
                densities_path=root / "probmap_densities.npz",
                density_method="gpu-hist",
                feature_batch=1,
                kde_points=32,
                kde_max_samples=32,
                kde_bandwidth=None,
                hist_sigma_bins=1.5,
                bg_prob_threshold=0.4,
                fg_prob_threshold=0.95,
                seed=1337,
                device_name="cpu",
            )

            densities_path = probability_map_script.run_probability_map(root, params=params)

            self.assertEqual(densities_path, root / "probmap_densities.npz")
            self.assertTrue(densities_path.is_file())
            for name in ("sample_a", "sample_b"):
                output_dir = root / name / "probmap"
                self.assertTrue((output_dir / "semantic_seg.tif").is_file())
                self.assertTrue((output_dir / "instance_seg.tif").is_file())
                self.assertTrue((output_dir / "probmap.tif").is_file())
                self.assertFalse((output_dir / "export.tif").exists())
                self.assertFalse((output_dir / "export2.tif").exists())
                semantic_seg = tifffile.imread(output_dir / "semantic_seg.tif")
                instance_seg = tifffile.imread(output_dir / "instance_seg.tif")
                probmap = tifffile.imread(output_dir / "probmap.tif")
                self.assertEqual(semantic_seg.shape, (4, 4, 4))
                self.assertEqual(instance_seg.shape, (4, 4, 4))
                self.assertEqual(probmap.shape, (4, 4, 4))
                np.testing.assert_array_equal(instance_seg > 0, semantic_seg > 0)


if __name__ == "__main__":
    unittest.main()
