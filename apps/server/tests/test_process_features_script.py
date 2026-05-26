from __future__ import annotations

import importlib.util
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch


def _load_process_features_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "post_processing" / "process_features.py"
    spec = importlib.util.spec_from_file_location("process_features_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load process-features module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


process_features_script = _load_process_features_module()


class ProcessFeaturesScriptTests(unittest.TestCase):
    def test_select_timepoints_uses_zero_based_exclusive_end(self) -> None:
        timepoints = [SimpleNamespace(name=name) for name in ("sample_a", "sample_b", "sample_c")]

        selected = process_features_script.select_timepoints(timepoints, file_start=1, file_end=3)

        self.assertEqual([timepoint.name for timepoint in selected], ["sample_b", "sample_c"])

    def test_select_timepoints_rejects_empty_selection(self) -> None:
        timepoints = [SimpleNamespace(name=name) for name in ("sample_a", "sample_b")]

        with self.assertRaisesRegex(ValueError, "zero timepoints"):
            process_features_script.select_timepoints(timepoints, file_start=1, file_end=1)

    def test_cleanup_output_root_only_removes_requested_pca_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hr_dir = root / "hr_feats" / "sample_a"
            pca_3_dir = root / "pca_3"
            pca_5_dir = root / "pca_5"
            hr_dir.mkdir(parents=True)
            pca_3_dir.mkdir()
            pca_5_dir.mkdir()
            (hr_dir / "feature_000.tif").write_bytes(b"hr")
            (pca_3_dir / "sample_a.tif").write_bytes(b"pca3")
            (pca_5_dir / "sample_a.tif").write_bytes(b"pca5")

            process_features_script.cleanup_output_root(
                root,
                save_pca=True,
                pca_components=3,
                save_high_resolution_features=False,
            )

            self.assertTrue((root / "hr_feats").exists())
            self.assertFalse((root / "pca_3").exists())
            self.assertTrue((root / "pca_5").exists())

    def test_cleanup_output_root_only_removes_high_resolution_features_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hr_dir = root / "hr_feats" / "sample_a"
            pca_3_dir = root / "pca_3"
            hr_dir.mkdir(parents=True)
            pca_3_dir.mkdir()
            (hr_dir / "feature_000.tif").write_bytes(b"hr")
            (pca_3_dir / "sample_a.tif").write_bytes(b"pca3")

            process_features_script.cleanup_output_root(
                root,
                save_pca=False,
                pca_components=3,
                save_high_resolution_features=True,
            )

            self.assertFalse((root / "hr_feats").exists())
            self.assertTrue((root / "pca_3").exists())

    def test_parse_bool_accepts_explicit_false(self) -> None:
        self.assertFalse(process_features_script.parse_bool("false"))
        self.assertTrue(process_features_script.parse_bool("true"))

    def test_parse_args_defaults_global_pca_to_true(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["process_features.py", "--input-path", "/tmp/input", "--save-pca"],
        ):
            args = process_features_script.parse_args()

        self.assertTrue(args.global_pca)

    def test_parse_args_accepts_global_pca_false(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["process_features.py", "--input-path", "/tmp/input", "--save-pca", "--global-pca", "false"],
        ):
            args = process_features_script.parse_args()

        self.assertFalse(args.global_pca)

    def test_global_pca_uses_shared_basis_and_ranges(self) -> None:
        first = np.array([[[[0.0, 0.0], [2.0, 0.0]]]], dtype=np.float32)
        second = np.array([[[[4.0, 0.0], [6.0, 0.0]]]], dtype=np.float32)
        device = torch.device("cpu")

        model = process_features_script.fit_pca_model_from_sources(
            [("first", first), ("second", second)],
            n_components=1,
            device=device,
        )
        mins, maxs = process_features_script.compute_global_pca_min_max_from_sources(
            [("first", first), ("second", second)],
            pca_model=model,
            device=device,
        )
        projected_first = process_features_script.project_pca_volume(first, pca_model=model, device=device)
        projected_second = process_features_script.project_pca_volume(second, pca_model=model, device=device)

        self.assertGreater(model.components[0, 0], 0.0)
        np.testing.assert_allclose(model.mean, np.array([3.0, 0.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(mins, np.array([-3.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(maxs, np.array([3.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(projected_first.reshape(-1), np.array([-3.0, -1.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(projected_second.reshape(-1), np.array([1.0, 3.0], dtype=np.float32), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
