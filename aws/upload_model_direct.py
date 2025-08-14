#!/usr/bin/env python3
"""
Direct model uploader to S3 - simplified approach based on existing system.
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List

import boto3
from tqdm import tqdm
from botocore.exceptions import NoCredentialsError, ClientError

# Configuration
BUCKET_NAME = "spatialdino"
S3_PREFIX = "models"
AWS_REGION = "us-east-1"
CHUNK_SIZE = 100 * 1024 * 1024  # 100MB chunks


class ModelUploader:
    """Simple but robust model uploader."""

    def __init__(self):
        self.setup_logging()
        self.setup_aws_client()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("model_upload.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_aws_client(self):
        """Setup AWS S3 client."""
        try:
            self.s3_client = boto3.client("s3", region_name=AWS_REGION)
            # Test connection
            self.s3_client.head_bucket(Bucket=BUCKET_NAME)
            self.logger.info(f"✓ Connected to S3 bucket: {BUCKET_NAME}")
        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"✗ AWS S3 connection failed: {e}")
            sys.exit(1)

    def find_files_to_upload(self, model_dir: str, model_name: str) -> List[Dict]:
        """Find all files in the model directory."""
        model_path = Path(model_dir)
        if not model_path.exists():
            self.logger.error(f"Model directory not found: {model_dir}")
            return []

        files_to_upload = []

        # Find all files recursively
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(model_path)
                s3_key = f"{S3_PREFIX}/{model_name}/{rel_path}"

                file_info = {
                    "local_path": str(file_path),
                    "s3_key": s3_key,
                    "size": file_path.stat().st_size,
                    "filename": file_path.name,
                }
                files_to_upload.append(file_info)

        self.logger.info(f"Found {len(files_to_upload)} files to upload")
        total_size = sum(f["size"] for f in files_to_upload)
        self.logger.info(f"Total size: {total_size / (1024**3):.2f} GB")

        return files_to_upload

    def calculate_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def check_if_uploaded(self, s3_key: str, local_size: int) -> bool:
        """Check if file already exists in S3 with same size."""
        try:
            response = self.s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
            s3_size = response.get("ContentLength", 0)
            return s3_size == local_size
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                self.logger.warning(f"Error checking {s3_key}: {e}")
            return False

    def upload_file(self, file_info: Dict) -> bool:
        """Upload a single file to S3."""
        try:
            local_path = file_info["local_path"]
            s3_key = file_info["s3_key"]

            self.logger.info(
                f"Uploading: {file_info['filename']} ({file_info['size'] / (1024**2):.1f} MB)"
            )

            # Calculate MD5 for integrity
            md5_hash = self.calculate_md5(local_path)

            # Upload file
            with open(local_path, "rb") as f:
                self.s3_client.put_object(
                    Bucket=BUCKET_NAME,
                    Key=s3_key,
                    Body=f,
                    Metadata={
                        "md5_hash": md5_hash,
                        "upload_time": str(int(time.time())),
                        "original_path": local_path,
                    },
                )

            self.logger.info(f"✓ Uploaded: {s3_key}")
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to upload {file_info['filename']}: {e}")
            return False

    def upload_model(self, model_dir: str, model_name: str):
        """Upload entire model directory."""
        self.logger.info(f"Starting upload for model: {model_name}")
        self.logger.info(f"Source directory: {model_dir}")

        # Find files
        files_to_upload = self.find_files_to_upload(model_dir, model_name)
        if not files_to_upload:
            self.logger.error("No files found to upload")
            return

        # Check which files already exist
        files_needed = []
        skipped_count = 0

        for file_info in files_to_upload:
            if self.check_if_uploaded(file_info["s3_key"], file_info["size"]):
                self.logger.info(f"⏭ Already exists: {file_info['filename']}")
                skipped_count += 1
            else:
                files_needed.append(file_info)

        self.logger.info(
            f"Files to upload: {len(files_needed)}, already exist: {skipped_count}"
        )

        if not files_needed:
            self.logger.info("All files already uploaded!")
            return

        # Upload files
        successful = 0
        failed = 0

        with tqdm(files_needed, desc="Uploading files") as pbar:
            for file_info in pbar:
                pbar.set_description(f"Uploading {file_info['filename']}")
                if self.upload_file(file_info):
                    successful += 1
                else:
                    failed += 1
                pbar.update()

        # Summary
        self.logger.info("=" * 50)
        self.logger.info("UPLOAD SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"✓ Successfully uploaded: {successful} files")
        self.logger.info(f"⏭ Already existed: {skipped_count} files")
        self.logger.info(f"✗ Failed: {failed} files")
        self.logger.info(
            f"🌐 S3 location: s3://{BUCKET_NAME}/{S3_PREFIX}/{model_name}/"
        )

        if failed == 0:
            self.logger.info("🎉 Upload completed successfully!")
        else:
            self.logger.warning(f"⚠️  Upload completed with {failed} failures")


def main():
    if len(sys.argv) != 3:
        print("Usage: python upload_model_direct.py <model_dir> <model_name>")
        print("Example: python upload_model_direct.py /path/to/model my-model-name")
        sys.exit(1)

    model_dir = sys.argv[1]
    model_name = sys.argv[2]

    uploader = ModelUploader()
    uploader.upload_model(model_dir, model_name)


if __name__ == "__main__":
    main()
