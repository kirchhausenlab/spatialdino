#!/usr/bin/env python3
"""
S3 Upload Summary - Shows current status and provides quick commands.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path


def format_bytes(bytes_val):
    """Format bytes as human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def show_progress():
    """Show current upload progress."""
    progress_file = "s3_upload_progress.json"

    if not os.path.exists(progress_file):
        print("❌ No progress file found. Upload hasn't started yet.")
        return

    try:
        with open(progress_file, "r") as f:
            data = json.load(f)

        files = data.get("files", [])
        timestamp = data.get("timestamp", 0)

        uploaded = [f for f in files if f.get("uploaded", False)]
        failed = [f for f in files if not f.get("uploaded", False)]

        total_files = len(files)
        uploaded_files = len(uploaded)
        failed_files = len(failed)

        total_size = sum(f.get("size", 0) for f in files)
        uploaded_size = sum(f.get("size", 0) for f in uploaded)

        progress_pct = (uploaded_files / total_files * 100) if total_files > 0 else 0
        size_pct = (uploaded_size / total_size * 100) if total_size > 0 else 0

        last_update = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        print("📊 UPLOAD PROGRESS")
        print("=" * 50)
        print(f"Files:     {uploaded_files:,} / {total_files:,} ({progress_pct:.1f}%)")
        print(
            f"Size:      {format_bytes(uploaded_size)} / {format_bytes(total_size)} ({size_pct:.1f}%)"
        )
        print(f"Failed:    {failed_files:,} files")
        print(f"Updated:   {last_update}")
        print("=" * 50)

        if failed_files > 0:
            print(f"⚠️  {failed_files} files failed to upload. You can resume to retry.")

        if uploaded_files == total_files and failed_files == 0:
            print("✅ All files uploaded successfully!")
        elif uploaded_files > 0:
            print("🔄 Upload in progress or paused. You can resume anytime.")

    except Exception as e:
        print(f"❌ Error reading progress file: {e}")


def show_dataset_info():
    """Show information about datasets to be uploaded."""
    try:
        from .s3_upload_config import get_config

        config = get_config()

        print("📁 DATASET INFORMATION")
        print("=" * 50)
        print(f"S3 Bucket:  {config.bucket_name}")
        print(f"S3 Prefix:  {config.s3_prefix}")
        print()

        total_files = 0
        total_size = 0

        for local_dir, dataset_name in config.dataset_mappings.items():
            if os.path.exists(local_dir):
                file_count = sum(1 for _, _, files in os.walk(local_dir) for _ in files)
                dir_size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, _, files in os.walk(local_dir)
                    for file in files
                )
                total_files += file_count
                total_size += dir_size

                print(f"{dataset_name}:")
                print(f"  Source:  {local_dir}")
                print(f"  Files:   {file_count:,}")
                print(f"  Size:    {format_bytes(dir_size)}")
                print()
            else:
                print(f"{dataset_name}: ❌ Directory not found: {local_dir}")
                print()

        print(f"TOTAL: {total_files:,} files, {format_bytes(total_size)}")
        print("=" * 50)

    except ImportError:
        print("❌ Configuration not available. Run setup first.")
    except Exception as e:
        print(f"❌ Error getting dataset info: {e}")


def show_quick_commands():
    """Show quick command reference."""
    print("🚀 QUICK COMMANDS")
    print("=" * 50)
    print("Setup:     ./run_s3_upload.sh setup")
    print("Test:      ./run_s3_upload.sh dry-run")
    print("Upload:    ./run_s3_upload.sh upload")
    print("Resume:    ./run_s3_upload.sh resume")
    print("Verify:    ./run_s3_upload.sh verify")
    print("Status:    ./run_s3_upload.sh status")
    print()
    print("Direct Python commands:")
    print("  python setup_s3_upload.py")
    print("  python s3_uploader.py --dry-run")
    print("  python s3_uploader.py")
    print("  python s3_verifier.py")
    print("=" * 50)


def main():
    """Main function."""
    print()
    print("🌟 S3 DATASET UPLOAD TOOL")
    print()

    show_dataset_info()
    print()
    show_progress()
    print()
    show_quick_commands()
    print()


if __name__ == "__main__":
    main()
