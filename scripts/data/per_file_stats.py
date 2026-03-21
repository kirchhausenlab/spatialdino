#!/usr/bin/env python3
"""
Prints per-annotation-file statistics:
- Total data size (GB)
- Average z-frames (TIFs) per experiment
- Average size per experiment (GB)

If a path in the annotation file is a directory, each immediate subdirectory is treated as a separate experiment.
Also saves a markdown file with the same stats.
"""

import os
from pathlib import Path


def analyze_folder(path):
    """Recursively compute the total size and TIF count under a directory.

    Args:
        path: Root directory to walk.

    Returns:
        Tuple of (total_size_bytes, tif_file_count).
    """
    total_size = 0
    tif_count = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                tif_count += 1
            try:
                total_size += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    return total_size, tif_count


def expand_experiments(path):
    """If path is a directory, return its immediate subdirectories as experiments. Else, return [path]."""
    if os.path.isdir(path):
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        # If no subdirs, treat the directory itself as an experiment
        return subdirs if subdirs else [path]
    else:
        return [path]


def main():
    """Compute and print per-annotation-file statistics.

    Reads every ``.txt`` file in the annotations directory, expands paths
    into experiments, gathers size/TIF counts, and prints a summary to
    stdout. Also writes a Markdown report to ``scripts/data/per_file_stats.md``.
    """
    annotations_dir = Path("/nfs/scratch2/shared_image_recog_ml/annotations")
    txt_files = sorted(annotations_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} annotation files.")
    md_lines = ["# Per-Annotation File Data Statistics\n"]
    for txt_file in txt_files:
        exp_paths = [line.strip() for line in open(txt_file) if line.strip()]
        all_experiments = []
        for exp_path in exp_paths:
            all_experiments.extend(expand_experiments(exp_path))
        exp_stats = []
        for exp_path in all_experiments:
            if os.path.isdir(exp_path):
                size, tifs = analyze_folder(exp_path)
            elif os.path.isfile(exp_path):
                size = os.path.getsize(exp_path)
                tifs = 1 if exp_path.lower().endswith((".tif", ".tiff")) else 0
            else:
                size, tifs = 0, 0
            exp_stats.append((size, tifs))
        if not exp_stats:
            continue
        total_size = sum(s for s, _ in exp_stats)
        total_tifs = sum(t for _, t in exp_stats)
        avg_tifs = total_tifs / len(exp_stats)
        avg_size = total_size / len(exp_stats) / (1024**3)
        print(f"\nFile: {txt_file.name}")
        print(f"  Experiments: {len(exp_stats)}")
        print(f"  Total data: {total_size / (1024**3):.2f} GB")
        print(f"  Avg z-frames (TIFs) per experiment: {avg_tifs:.1f}")
        print(f"  Avg size per experiment: {avg_size:.2f} GB")
        if txt_file.name in ("dgx1.txt", "dgx2.txt"):
            print("  (This is a DGX file)")
        md_lines.append(f"## File: {txt_file.name}\n")
        md_lines.append(f"- Experiments: {len(exp_stats)}\n")
        md_lines.append(f"- Total data: {total_size / (1024**3):.2f} GB\n")
        md_lines.append(f"- Avg z-frames (TIFs) per experiment: {avg_tifs:.1f}\n")
        md_lines.append(f"- Avg size per experiment: {avg_size:.2f} GB\n")
        if txt_file.name in ("dgx1.txt", "dgx2.txt"):
            md_lines.append(f"- (This is a DGX file)\n")
    with open("scripts/data/per_file_stats.md", "w") as f:
        f.write("\n".join(md_lines))
    print("\nMarkdown summary saved to scripts/data/per_file_stats.md")


if __name__ == "__main__":
    main()
