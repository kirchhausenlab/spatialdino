#!/usr/bin/env python3
"""
Efficient S3 uploader for large datasets with progress tracking and resume capability.
"""

import os
import sys
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import boto3
import psutil
from tqdm import tqdm
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError

from .s3_upload_config import get_config


@dataclass
class FileInfo:
    """Information about a file to upload."""

    local_path: str
    s3_key: str
    size: int
    md5_hash: Optional[str] = None
    uploaded: bool = False
    upload_id: Optional[str] = None
    parts: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.parts is None:
            self.parts = []


class S3Uploader:
    """High-performance S3 uploader with resume capability."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.setup_logging()
        self.setup_aws_client()
        self.progress_data = self.load_progress()
        self.bandwidth_limiter = BandwidthLimiter(self.config.max_bandwidth_mbps)

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("s3_upload.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_aws_client(self):
        """Setup AWS S3 client with optimizations."""
        try:
            # Configure for high-throughput operations
            config = Config(
                region_name=self.config.aws_region,
                retries={"max_attempts": self.config.max_retries},
                max_pool_connections=50,
                read_timeout=300,
                connect_timeout=300,
            )

            self.s3_client = boto3.client("s3", config=config)

            # Test credentials
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
            self.logger.info(
                f"Successfully connected to S3 bucket: {self.config.bucket_name}"
            )

        except NoCredentialsError:
            self.logger.error(
                "AWS credentials not found. Please configure AWS credentials."
            )
            sys.exit(1)
        except ClientError as e:
            self.logger.error(f"Error accessing S3 bucket: {e}")
            sys.exit(1)

    def scan_files(self) -> List[FileInfo]:
        """Scan all directories and collect file information."""
        self.logger.info("Scanning directories for files...")
        files = []

        for local_dir, dataset_name in self.config.dataset_mappings.items():
            if not os.path.exists(local_dir):
                self.logger.warning(f"Directory not found: {local_dir}")
                continue

            self.logger.info(f"Scanning {local_dir} -> {dataset_name}")

            for root, dirs, filenames in os.walk(local_dir):
                for filename in filenames:
                    local_path = os.path.join(root, filename)

                    # Create S3 key
                    rel_path = os.path.relpath(local_path, local_dir)
                    s3_key = f"{self.config.s3_prefix}/{dataset_name}/{rel_path}"

                    # Get file size
                    file_size = os.path.getsize(local_path)

                    files.append(
                        FileInfo(local_path=local_path, s3_key=s3_key, size=file_size)
                    )

        total_size = sum(f.size for f in files)
        self.logger.info(
            f"Found {len(files)} files, total size: {self._format_bytes(total_size)}"
        )

        return files

    def load_progress(self) -> Dict:
        """Load progress from previous run."""
        try:
            if os.path.exists(self.config.progress_file):
                with open(self.config.progress_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load progress file: {e}")
        return {}

    def save_progress(self, files: List[FileInfo]):
        """Save current progress."""
        progress = {"files": [asdict(f) for f in files], "timestamp": time.time()}

        try:
            with open(self.config.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")

    def check_existing_files(self, files: List[FileInfo]) -> List[FileInfo]:
        """Check which files already exist in S3 and filter them out."""
        self.logger.info("Checking for existing files in S3...")

        files_to_upload = []
        skipped_count = 0

        with tqdm(total=len(files), desc="Checking S3") as pbar:
            for file_info in files:
                try:
                    response = self.s3_client.head_object(
                        Bucket=self.config.bucket_name, Key=file_info.s3_key
                    )

                    # Check if size matches
                    s3_size = response.get("ContentLength", 0)
                    if s3_size == file_info.size:
                        self.logger.debug(f"Skipping existing file: {file_info.s3_key}")
                        file_info.uploaded = True
                        skipped_count += 1
                    else:
                        files_to_upload.append(file_info)

                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        # File doesn't exist, add to upload list
                        files_to_upload.append(file_info)
                    else:
                        self.logger.error(f"Error checking {file_info.s3_key}: {e}")
                        files_to_upload.append(file_info)

                pbar.update(1)

        self.logger.info(
            f"Skipped {skipped_count} existing files, {len(files_to_upload)} files to upload"
        )
        return files_to_upload

    def upload_file(self, file_info: FileInfo) -> bool:
        """Upload a single file with multipart upload for large files."""
        try:
            if file_info.size < self.config.chunk_size:
                return self._upload_small_file(file_info)
            else:
                return self._upload_large_file(file_info)

        except Exception as e:
            self.logger.error(f"Error uploading {file_info.local_path}: {e}")
            return False

    def _upload_small_file(self, file_info: FileInfo) -> bool:
        """Upload small file in a single operation."""
        try:
            with open(file_info.local_path, "rb") as f:
                data = f.read()

            self.bandwidth_limiter.wait(len(data))

            self.s3_client.put_object(
                Bucket=self.config.bucket_name, Key=file_info.s3_key, Body=data
            )

            file_info.uploaded = True
            return True

        except Exception as e:
            self.logger.error(f"Error uploading small file {file_info.local_path}: {e}")
            return False

    def _upload_large_file(self, file_info: FileInfo) -> bool:
        """Upload large file using multipart upload."""
        try:
            # Start multipart upload
            if not file_info.upload_id:
                response = self.s3_client.create_multipart_upload(
                    Bucket=self.config.bucket_name, Key=file_info.s3_key
                )
                file_info.upload_id = response["UploadId"]

            # Upload parts
            part_number = len(file_info.parts) + 1

            with open(file_info.local_path, "rb") as f:
                f.seek((part_number - 1) * self.config.chunk_size)

                while True:
                    data = f.read(self.config.chunk_size)
                    if not data:
                        break

                    self.bandwidth_limiter.wait(len(data))

                    response = self.s3_client.upload_part(
                        Bucket=self.config.bucket_name,
                        Key=file_info.s3_key,
                        PartNumber=part_number,
                        UploadId=file_info.upload_id,
                        Body=data,
                    )

                    file_info.parts.append({
                        "ETag": response["ETag"],
                        "PartNumber": part_number,
                    })

                    part_number += 1

            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=file_info.s3_key,
                UploadId=file_info.upload_id,
                MultipartUpload={"Parts": file_info.parts},
            )

            file_info.uploaded = True
            return True

        except Exception as e:
            self.logger.error(f"Error uploading large file {file_info.local_path}: {e}")
            # Don't abort multipart upload here - allow resume
            return False

    def upload_files(self, files: List[FileInfo]):
        """Upload all files with parallel processing."""
        if not files:
            self.logger.info("No files to upload")
            return

        total_size = sum(f.size for f in files if not f.uploaded)
        self.logger.info(
            f"Starting upload of {len(files)} files ({self._format_bytes(total_size)})"
        )

        # Create progress bar
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Uploading")

        # Monitor system resources
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()

        # Upload files in parallel
        successful_uploads = 0
        failed_uploads = 0

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(self.upload_file, file_info): file_info
                for file_info in files
                if not file_info.uploaded
            }

            # Process completed uploads
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]

                try:
                    success = future.result()
                    if success:
                        successful_uploads += 1
                        pbar.update(file_info.size)
                        self.logger.debug(f"✓ Uploaded: {file_info.s3_key}")
                    else:
                        failed_uploads += 1
                        self.logger.error(f"✗ Failed: {file_info.s3_key}")

                except Exception as e:
                    failed_uploads += 1
                    self.logger.error(f"✗ Exception uploading {file_info.s3_key}: {e}")

                # Save progress periodically
                if (successful_uploads + failed_uploads) % 10 == 0:
                    self.save_progress(files)

        pbar.close()

        # Final progress save
        self.save_progress(files)

        self.logger.info(
            f"Upload completed: {successful_uploads} successful, {failed_uploads} failed"
        )

        if failed_uploads > 0:
            self.logger.warning(
                f"{failed_uploads} files failed to upload. You can re-run to retry."
            )

    def _monitor_resources(self):
        """Monitor system resources during upload."""
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()

            if cpu_percent > 90 or memory.percent > 90:
                self.logger.warning(
                    f"High resource usage: CPU {cpu_percent}%, RAM {memory.percent}%"
                )

            time.sleep(30)

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes as human readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"

    def run(self):
        """Main upload process."""
        files = []
        try:
            # Scan files
            files = self.scan_files()

            # Check existing files
            files_to_upload = self.check_existing_files(files)

            # Upload files
            self.upload_files(files_to_upload)

        except KeyboardInterrupt:
            self.logger.info("Upload interrupted by user")
            if files:
                self.save_progress(files)
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            if files:
                self.save_progress(files)
            raise


class BandwidthLimiter:
    """Simple bandwidth limiter."""

    def __init__(self, max_mbps: Optional[int]):
        self.max_bytes_per_second = max_mbps * 1024 * 1024 if max_mbps else None
        self.last_time = time.time()
        self.bytes_sent = 0

    def wait(self, bytes_to_send: int):
        """Wait if necessary to respect bandwidth limit."""
        if not self.max_bytes_per_second:
            return

        self.bytes_sent += bytes_to_send
        current_time = time.time()
        elapsed = current_time - self.last_time

        if elapsed >= 1.0:
            # Reset counter every second
            self.last_time = current_time
            self.bytes_sent = 0
        elif self.bytes_sent > self.max_bytes_per_second:
            # Need to wait
            wait_time = 1.0 - elapsed
            time.sleep(wait_time)
            self.last_time = time.time()
            self.bytes_sent = 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload large datasets to S3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous upload"
    )
    parser.add_argument("--config", help="Path to custom config file")

    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Create uploader
    uploader = S3Uploader(config)

    if args.dry_run:
        files = uploader.scan_files()
        total_size = sum(f.size for f in files)
        print(f"Would upload {len(files)} files ({uploader._format_bytes(total_size)})")
        dataset_counts = {}
        for f in files:
            dataset = f.s3_key.split("/")[1] if "/" in f.s3_key else "unknown"
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} files")
    else:
        uploader.run()


if __name__ == "__main__":
    main()
