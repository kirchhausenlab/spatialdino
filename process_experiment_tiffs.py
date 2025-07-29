#!/usr/bin/env python3

import shutil
from pathlib import Path
import argparse


def process_experiments(input_path, output_path, pattern="Ex*"):
    """
    Process experiment folders in input_path and copy volume_unnorm.tif files to output_path
    with renamed structure.

    Args:
        input_path: Path containing experiment folders
        output_path: Destination path for renamed tif files
        pattern: Pattern to match experiment folders (default: "Ex*")
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Try multiple patterns if default doesn't work
    patterns = [pattern, "ex*", "Ex*", "*Ex*", "*ex*"]
    ex_folders = []

    for p in patterns:
        ex_folders = list(input_path.glob(p))
        if ex_folders:
            print(f"Found experiment folders using pattern '{p}'")
            break

    if not ex_folders:
        print(f"No experiment folders found in {input_path} using patterns {patterns}")
        return

    print(f"Found {len(ex_folders)} experiment folders in {input_path}")

    processed = 0
    for ex_folder in sorted(ex_folders):
        if ex_folder.is_dir():
            volume_file = ex_folder / "volume_unnorm.tif"

            if volume_file.exists():
                # Create output filename: {folder_name}.tif
                output_file = output_path / f"{ex_folder.name}.tif"

                # Copy and rename the file
                shutil.copy2(volume_file, output_file)
                print(
                    f"Copied: {ex_folder.name}/volume_unnorm.tif -> {output_file.name}"
                )
                processed += 1
            else:
                print(f"Warning: volume_unnorm.tif not found in {ex_folder}")

    print(f"Successfully processed {processed} files to {output_path}")


def main():
    """Main function to process all specified paths."""

    # Define all the paths to process
    paths_to_process = [
        {
            "input": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/ch488nmCamA/DS/zeiss_488_left",
            "output": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/ch488nmCamA/DS",
            "pattern": "Ex*",
        },
        {
            "input": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/ch560nmCamB/DS/zeiss_560_left",
            "output": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/ch560nmCamB/DS",
            "pattern": "Ex*",
        },
        {
            "input": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/ch642nmCamA/DS/zeiss_642_left",
            "output": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/ch642nmCamA/DS",
            "pattern": "Ex*",
        },
        {
            "input": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/ap2_unnorm_new/ch488nmCamA/DS",
            "output": "/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/ap2_unnorm_new/ch488nmCamA/DS",
            "pattern": "ex*",
        },
    ]

    for i, path_config in enumerate(paths_to_process, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing path {i}/{len(paths_to_process)}")
        print(f"Input:  {path_config['input']}")
        print(f"Output: {path_config['output']}")
        print(f"Pattern: {path_config['pattern']}")
        print(f"{'=' * 60}")

        if not Path(path_config["input"]).exists():
            print(f"Warning: Input path does not exist: {path_config['input']}")
            continue

        process_experiments(
            path_config["input"], path_config["output"], path_config["pattern"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment tiff files")
    parser.add_argument(
        "--input", help="Custom input path containing experiment folders"
    )
    parser.add_argument("--output", help="Custom output path for renamed tif files")
    parser.add_argument(
        "--pattern",
        default="Ex*",
        help="Pattern to match experiment folders (default: Ex*)",
    )

    args = parser.parse_args()

    if args.input and args.output:
        # Process custom paths
        process_experiments(args.input, args.output, args.pattern)
    else:
        # Process all predefined paths
        main()

    """
snr4_data 
python3 process_experiment_tiffs.py \
    --input "/raid1/cme_tests/results/simulated/snr4_high" \
    --output "/raid1/cme_tests/results/simulated/snr4_high_processed" \
    --pattern "VIRUS_*"

Standard
python3 process_experiment_tiffs.py \
    --input "/path/to/ex/folders" \
    --output "/path/to/output" \
    --pattern "Ex*"
    
    """
