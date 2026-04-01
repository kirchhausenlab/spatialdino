from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
