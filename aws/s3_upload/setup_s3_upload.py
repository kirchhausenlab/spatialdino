#!/usr/bin/env python3
"""
Setup script for S3 upload environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("\nChecking AWS credentials...")

    # Check environment variables
    aws_keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    env_creds = all(os.getenv(key) for key in aws_keys)

    # Check AWS CLI config
    aws_config_path = Path.home() / ".aws" / "credentials"
    config_creds = aws_config_path.exists()

    if env_creds:
        print("✓ AWS credentials found in environment variables")
        return True
    elif config_creds:
        print("✓ AWS credentials found in ~/.aws/credentials")
        return True
    else:
        print("✗ AWS credentials not found")
        print("\nTo configure AWS credentials, you can:")
        print("1. Set environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   export AWS_DEFAULT_REGION=us-east-1")
        print("\n2. Or run: aws configure")
        return False


def install_dependencies():
    """Install required Python packages."""
    requirements_file = "requirements_s3.txt"

    if not os.path.exists(requirements_file):
        print(f"✗ Requirements file {requirements_file} not found")
        return False

    # Try pip install
    cmd = f"pip install -r {requirements_file}"
    if run_command(cmd, "Installing Python dependencies"):
        return True

    # Try conda install if pip fails
    print("Trying conda install as fallback...")
    packages = ["boto3", "tqdm", "python-dotenv", "psutil"]
    cmd = f"conda install -y {' '.join(packages)}"
    return run_command(cmd, "Installing dependencies via conda")


def check_directories():
    """Check if source directories exist."""
    from s3_upload_config import get_config

    config = get_config()
    print("\nChecking source directories...")

    all_exist = True
    for local_dir, dataset_name in config.dataset_mappings.items():
        if os.path.exists(local_dir):
            file_count = sum(1 for _, _, files in os.walk(local_dir) for _ in files)
            total_size = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(local_dir)
                for file in files
            )
            size_gb = total_size / (1024**3)
            print(
                f"✓ {local_dir} -> {dataset_name}: {file_count} files, {size_gb:.1f} GB"
            )
        else:
            print(f"✗ {local_dir} -> {dataset_name}: Directory not found")
            all_exist = False

    return all_exist


def test_s3_connection():
    """Test S3 connection and bucket access."""
    try:
        import boto3
        from .s3_upload_config import get_config

        config = get_config()
        s3_client = boto3.client("s3", region_name=config.aws_region)

        print(f"\nTesting S3 connection to bucket: {config.bucket_name}")
        s3_client.head_bucket(Bucket=config.bucket_name)
        print("✓ S3 connection successful")
        return True

    except Exception as e:
        print(f"✗ S3 connection failed: {e}")
        return False


def main():
    """Main setup process."""
    print("=" * 60)
    print("S3 UPLOAD SETUP")
    print("=" * 60)

    success = True

    # Install dependencies
    if not install_dependencies():
        success = False

    # Check AWS credentials
    if not check_aws_credentials():
        success = False

    # Check directories
    if not check_directories():
        success = False

    # Test S3 connection (only if other checks pass)
    if success and not test_s3_connection():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ SETUP COMPLETE - Ready to upload!")
        print("\nNext steps:")
        print("1. Test with dry run: python s3_uploader.py --dry-run")
        print("2. Start upload: python s3_uploader.py")
        print("3. Verify upload: python s3_verifier.py")
    else:
        print("✗ SETUP INCOMPLETE - Please fix the issues above")
        return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
