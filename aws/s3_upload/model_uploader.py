#!/usr/bin/env python3
"""
Model weights uploader for S3 storage.
Handles uploading trained model weights to S3 bucket.
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import boto3
from tqdm import tqdm
from botocore.exceptions import ClientError, NoCredentialsError

from .s3_upload_config import get_config


@dataclass
class ModelInfo:
    """Information about a model to upload."""

    local_path: str
    model_name: str
    model_type: str  # 'backbone', 'checkpoint', 'config'
    s3_key: str
    size: int
    md5_hash: Optional[str] = None
    uploaded: bool = False


class ModelUploader:
    """Upload trained model weights to S3."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.setup_logging()
        self.setup_aws_client()

        # Model-specific S3 prefix
        self.models_prefix = f"{self.config.s3_prefix}/models"

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("model_upload.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_aws_client(self):
        """Setup AWS S3 client."""
        try:
            self.s3_client = boto3.client("s3", region_name=self.config.aws_region)
            # Test connection
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
            self.logger.info(f"Connected to S3 bucket: {self.config.bucket_name}")
        except (NoCredentialsError, ClientError) as e:
            self.logger.error(f"AWS S3 connection failed: {e}")
            sys.exit(1)

    def find_model_files(self, model_dir: str, model_name: str) -> List[ModelInfo]:
        """Find all model-related files in directory."""
        if not os.path.exists(model_dir):
            self.logger.error(f"Model directory not found: {model_dir}")
            return []

        model_files = []
        model_dir_path = Path(model_dir)

        # Define file patterns and types
        file_patterns = {
            "backbone.pth": "backbone",
            "*.pth": "checkpoint",
            "*.pt": "checkpoint",
            "*.ckpt": "checkpoint",
            "*.yaml": "config",
            "*.yml": "config",
            "*.json": "config",
            "*.txt": "metadata",
        }

        # Search for model files
        for pattern, file_type in file_patterns.items():
            if pattern.startswith("*"):
                # Glob pattern
                files = list(model_dir_path.glob(pattern))
            else:
                # Exact filename
                files = (
                    [model_dir_path / pattern]
                    if (model_dir_path / pattern).exists()
                    else []
                )

            for file_path in files:
                if file_path.is_file():
                    rel_path = file_path.relative_to(model_dir_path)
                    s3_key = f"{self.models_prefix}/{model_name}/{rel_path}"

                    model_files.append(
                        ModelInfo(
                            local_path=str(file_path),
                            model_name=model_name,
                            model_type=file_type,
                            s3_key=s3_key,
                            size=file_path.stat().st_size,
                        )
                    )

        self.logger.info(f"Found {len(model_files)} model files for {model_name}")
        return model_files

    def calculate_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def check_existing_model(self, model_info: ModelInfo) -> bool:
        """Check if model already exists in S3."""
        try:
            response = self.s3_client.head_object(
                Bucket=self.config.bucket_name, Key=model_info.s3_key
            )

            # Check size match
            s3_size = response.get("ContentLength", 0)
            if s3_size == model_info.size:
                self.logger.debug(f"Model already exists: {model_info.s3_key}")
                return True

        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                self.logger.warning(f"Error checking {model_info.s3_key}: {e}")

        return False

    def upload_model_file(self, model_info: ModelInfo) -> bool:
        """Upload a single model file."""
        try:
            self.logger.info(
                f"Uploading {model_info.model_type}: {model_info.local_path}"
            )

            # Calculate MD5 for integrity check
            model_info.md5_hash = self.calculate_md5(model_info.local_path)

            # Upload file
            with open(model_info.local_path, "rb") as f:
                self.s3_client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=model_info.s3_key,
                    Body=f,
                    Metadata={
                        "model_name": model_info.model_name,
                        "model_type": model_info.model_type,
                        "md5_hash": model_info.md5_hash,
                        "upload_time": str(int(time.time())),
                    },
                )

            model_info.uploaded = True
            self.logger.info(f"✓ Uploaded: {model_info.s3_key}")
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to upload {model_info.local_path}: {e}")
            return False

    def upload_model(self, model_dir: str, model_name: str) -> Dict:
        """Upload all files for a specific model."""
        self.logger.info(f"Starting upload for model: {model_name}")

        # Find model files
        model_files = self.find_model_files(model_dir, model_name)
        if not model_files:
            self.logger.warning(f"No model files found in {model_dir}")
            return {"uploaded": 0, "skipped": 0, "failed": 0}

        # Check existing files
        files_to_upload = []
        skipped_count = 0

        for model_info in model_files:
            if self.check_existing_model(model_info):
                model_info.uploaded = True
                skipped_count += 1
            else:
                files_to_upload.append(model_info)

        self.logger.info(
            f"Files to upload: {len(files_to_upload)}, already exist: {skipped_count}"
        )

        # Upload files
        successful_uploads = 0
        failed_uploads = 0

        for model_info in tqdm(files_to_upload, desc="Uploading model files"):
            if self.upload_model_file(model_info):
                successful_uploads += 1
            else:
                failed_uploads += 1

        # Create model manifest
        if successful_uploads > 0 or skipped_count > 0:
            self.create_model_manifest(model_name, model_files)

        return {
            "uploaded": successful_uploads,
            "skipped": skipped_count,
            "failed": failed_uploads,
        }

    def create_model_manifest(self, model_name: str, model_files: List[ModelInfo]):
        """Create a manifest file for the uploaded model."""
        manifest = {"model_name": model_name, "upload_time": time.time(), "files": []}

        for model_info in model_files:
            if model_info.uploaded:
                manifest["files"].append({
                    "filename": os.path.basename(model_info.local_path),
                    "s3_key": model_info.s3_key,
                    "size": model_info.size,
                    "type": model_info.model_type,
                    "md5_hash": model_info.md5_hash,
                })

        # Upload manifest
        manifest_key = f"{self.models_prefix}/{model_name}/manifest.json"
        try:
            self.s3_client.put_object(
                Bucket=self.config.bucket_name,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json",
            )
            self.logger.info(f"✓ Created model manifest: {manifest_key}")
        except Exception as e:
            self.logger.warning(f"Failed to create manifest: {e}")

    def list_uploaded_models(self) -> List[str]:
        """List all models uploaded to S3."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            model_names = set()

            for page in paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=f"{self.models_prefix}/",
                Delimiter="/",
            ):
                # Get model directories
                for common_prefix in page.get("CommonPrefixes", []):
                    prefix = common_prefix["Prefix"]
                    model_name = prefix.split("/")[-2]  # Extract model name
                    model_names.add(model_name)

            return sorted(list(model_names))

        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []

    def download_model(self, model_name: str, download_dir: str) -> bool:
        """Download a model from S3."""
        download_path = Path(download_dir) / model_name
        download_path.mkdir(parents=True, exist_ok=True)

        try:
            # List all files for this model
            paginator = self.s3_client.get_paginator("list_objects_v2")
            files_downloaded = 0

            for page in paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=f"{self.models_prefix}/{model_name}/",
            ):
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    filename = os.path.basename(s3_key)
                    local_path = download_path / filename

                    self.logger.info(f"Downloading {filename}...")
                    self.s3_client.download_file(
                        self.config.bucket_name, s3_key, str(local_path)
                    )
                    files_downloaded += 1

            self.logger.info(
                f"✓ Downloaded {files_downloaded} files to {download_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error downloading model {model_name}: {e}")
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload/download model weights to/from S3"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload model to S3")
    upload_parser.add_argument("model_dir", help="Directory containing model files")
    upload_parser.add_argument("model_name", help="Name for the model in S3")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download model from S3")
    download_parser.add_argument("model_name", help="Name of model to download")
    download_parser.add_argument("download_dir", help="Directory to download to")

    # List command
    list_parser = subparsers.add_parser("list", help="List uploaded models")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    uploader = ModelUploader()

    if args.command == "upload":
        result = uploader.upload_model(args.model_dir, args.model_name)
        print(f"\nUpload Summary:")
        print(f"  Uploaded: {result['uploaded']} files")
        print(f"  Skipped: {result['skipped']} files")
        print(f"  Failed: {result['failed']} files")

    elif args.command == "download":
        success = uploader.download_model(args.model_name, args.download_dir)
        sys.exit(0 if success else 1)

    elif args.command == "list":
        models = uploader.list_uploaded_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models found in S3")


if __name__ == "__main__":
    main()
