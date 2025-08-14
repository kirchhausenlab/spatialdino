#!/usr/bin/env python3
"""
S3 upload verification script to ensure data integrity.
"""

import os
import sys
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import boto3
from tqdm import tqdm
from botocore.exceptions import ClientError

from .s3_upload_config import get_config


class S3Verifier:
    """Verify uploaded files match local files."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.setup_logging()
        self.setup_aws_client()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def setup_aws_client(self):
        """Setup AWS S3 client."""
        self.s3_client = boto3.client("s3", region_name=self.config.aws_region)

    def get_file_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of local file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def verify_file(self, local_path: str, s3_key: str) -> Tuple[bool, str]:
        """Verify a single file matches between local and S3."""
        try:
            # Get S3 object info
            response = self.s3_client.head_object(
                Bucket=self.config.bucket_name, Key=s3_key
            )

            # Check file size
            local_size = os.path.getsize(local_path)
            s3_size = response.get("ContentLength", 0)

            if local_size != s3_size:
                return False, f"Size mismatch: local={local_size}, s3={s3_size}"

            # Check ETag (MD5 for single-part uploads)
            s3_etag = response.get("ETag", "").strip('"')

            if "-" not in s3_etag:  # Single-part upload
                local_md5 = self.get_file_md5(local_path)
                if local_md5 != s3_etag:
                    return False, f"MD5 mismatch: local={local_md5}, s3={s3_etag}"
            else:
                # Multi-part upload - can't easily verify MD5
                self.logger.debug(f"Skipping MD5 check for multipart upload: {s3_key}")

            return True, "OK"

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False, "File not found in S3"
            else:
                return False, f"S3 error: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def scan_and_verify(self) -> Dict[str, List]:
        """Scan all expected files and verify them."""
        results = {"verified": [], "failed": [], "missing": []}

        self.logger.info("Scanning local files and verifying against S3...")

        # Collect all files to verify
        files_to_verify = []
        for local_dir, dataset_name in self.config.dataset_mappings.items():
            if not os.path.exists(local_dir):
                self.logger.warning(f"Directory not found: {local_dir}")
                continue

            for root, dirs, filenames in os.walk(local_dir):
                for filename in filenames:
                    local_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(local_path, local_dir)
                    s3_key = f"{self.config.s3_prefix}/{dataset_name}/{rel_path}"
                    files_to_verify.append((local_path, s3_key))

        self.logger.info(f"Found {len(files_to_verify)} files to verify")

        # Verify files in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(self.verify_file, local_path, s3_key): (
                    local_path,
                    s3_key,
                )
                for local_path, s3_key in files_to_verify
            }

            with tqdm(total=len(files_to_verify), desc="Verifying") as pbar:
                for future in as_completed(future_to_file):
                    local_path, s3_key = future_to_file[future]

                    try:
                        success, message = future.result()

                        if success:
                            results["verified"].append(s3_key)
                        else:
                            results["failed"].append((s3_key, message))
                            self.logger.error(f"✗ {s3_key}: {message}")

                    except Exception as e:
                        results["failed"].append((s3_key, str(e)))
                        self.logger.error(f"✗ {s3_key}: Exception - {e}")

                    pbar.update(1)

        return results

    def list_s3_files(self) -> List[str]:
        """List all files in S3 bucket with the given prefix."""
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.config.bucket_name, Prefix=self.config.s3_prefix
        ):
            if "Contents" in page:
                files.extend([obj["Key"] for obj in page["Contents"]])

        return files

    def find_orphaned_files(self) -> List[str]:
        """Find files in S3 that don't exist locally."""
        self.logger.info("Checking for orphaned files in S3...")

        # Get all S3 files
        s3_files = set(self.list_s3_files())

        # Get all expected local files
        expected_files = set()
        for local_dir, dataset_name in self.config.dataset_mappings.items():
            if not os.path.exists(local_dir):
                continue

            for root, dirs, filenames in os.walk(local_dir):
                for filename in filenames:
                    local_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(local_path, local_dir)
                    s3_key = f"{self.config.s3_prefix}/{dataset_name}/{rel_path}"
                    expected_files.add(s3_key)

        # Find orphaned files
        orphaned = s3_files - expected_files

        if orphaned:
            self.logger.warning(f"Found {len(orphaned)} orphaned files in S3")
            for file in sorted(orphaned):
                self.logger.warning(f"Orphaned: {file}")
        else:
            self.logger.info("No orphaned files found")

        return list(orphaned)

    def run_verification(self):
        """Run full verification process."""
        self.logger.info("Starting S3 upload verification...")

        # Verify uploaded files
        results = self.scan_and_verify()

        # Check for orphaned files
        orphaned = self.find_orphaned_files()

        # Print summary
        total_files = len(results["verified"]) + len(results["failed"])
        success_rate = (
            (len(results["verified"]) / total_files * 100) if total_files > 0 else 0
        )

        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total files checked: {total_files}")
        print(
            f"Successfully verified: {len(results['verified'])} ({success_rate:.1f}%)"
        )
        print(f"Failed verification: {len(results['failed'])}")
        print(f"Orphaned files in S3: {len(orphaned)}")

        if results["failed"]:
            print("\nFAILED FILES:")
            for s3_key, reason in results["failed"]:
                print(f"  ✗ {s3_key}: {reason}")

        if orphaned:
            print("\nORPHANED FILES (first 10):")
            for file in sorted(orphaned)[:10]:
                print(f"  ? {file}")
            if len(orphaned) > 10:
                print(f"  ... and {len(orphaned) - 10} more")

        print("=" * 60)

        return len(results["failed"]) == 0 and len(orphaned) == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify S3 upload integrity")
    parser.add_argument(
        "--orphaned-only", action="store_true", help="Only check for orphaned files"
    )
    args = parser.parse_args()

    verifier = S3Verifier()

    if args.orphaned_only:
        verifier.find_orphaned_files()
    else:
        success = verifier.run_verification()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
