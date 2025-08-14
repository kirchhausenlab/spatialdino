#!/usr/bin/env python3
"""
Comprehensive AWS CLI for Cell Interactome data and model management.
Handles uploading, downloading, and managing data and models on S3.
"""

import os
import sys
import json
import click
from pathlib import Path
from typing import List, Optional

# Add the aws directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from s3_upload.s3_uploader import S3Uploader
from s3_upload.s3_verifier import S3Verifier
from s3_upload.model_uploader import ModelUploader
from s3_upload.s3_upload_config import get_config


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Cell Interactome AWS CLI - Manage data and models on S3.

    This tool provides comprehensive functionality to:
    - Upload/download datasets and model weights
    - Verify data integrity
    - Manage S3 storage
    - Monitor transfer progress
    """
    pass


@cli.group()
def data():
    """Manage dataset uploads and downloads."""
    pass


@cli.group()
def models():
    """Manage model weights and checkpoints."""
    pass


@cli.group()
def storage():
    """Manage S3 storage and operations."""
    pass


# =============================================================================
# DATA COMMANDS
# =============================================================================


@data.command()
@click.option(
    "--dry-run", is_flag=True, help="Show what would be uploaded without uploading"
)
@click.option("--resume", is_flag=True, help="Resume interrupted upload")
def upload(dry_run: bool, resume: bool):
    """Upload datasets to S3."""
    config = get_config()
    uploader = S3Uploader(config)

    if dry_run:
        files = uploader.scan_files()
        total_size = sum(f.size for f in files)
        click.echo(
            f"Would upload {len(files)} files ({uploader._format_bytes(total_size)})"
        )

        dataset_counts = {}
        for f in files:
            dataset = f.s3_key.split("/")[1] if "/" in f.s3_key else "unknown"
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        for dataset, count in dataset_counts.items():
            click.echo(f"  {dataset}: {count} files")
    else:
        click.echo("Starting dataset upload...")
        uploader.run()


@data.command()
def verify():
    """Verify uploaded data integrity."""
    verifier = S3Verifier()
    success = verifier.run_verification()

    if success:
        click.echo("✅ All data verified successfully!")
    else:
        click.echo("❌ Verification found issues. Check logs for details.")
        sys.exit(1)


@data.command()
@click.argument("s3_prefix")
@click.argument("local_dir")
@click.option("--include", multiple=True, help="File patterns to include")
@click.option("--exclude", multiple=True, help="File patterns to exclude")
def download(s3_prefix: str, local_dir: str, include: tuple, exclude: tuple):
    """Download data from S3 to local directory."""
    config = get_config()

    try:
        import boto3

        s3_client = boto3.client("s3", region_name=config.aws_region)

        # Create local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # List and download files
        paginator = s3_client.get_paginator("list_objects_v2")
        files_downloaded = 0

        click.echo(
            f"Downloading from s3://{config.bucket_name}/{s3_prefix}/ to {local_dir}/"
        )

        for page in paginator.paginate(Bucket=config.bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]

                # Apply include/exclude filters
                if include and not any(pattern in s3_key for pattern in include):
                    continue
                if exclude and any(pattern in s3_key for pattern in exclude):
                    continue

                # Download file
                relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                local_path = Path(local_dir) / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                click.echo(f"Downloading {os.path.basename(s3_key)}...")
                s3_client.download_file(config.bucket_name, s3_key, str(local_path))
                files_downloaded += 1

        click.echo(f"✅ Downloaded {files_downloaded} files")

    except Exception as e:
        click.echo(f"❌ Download failed: {e}")
        sys.exit(1)


@data.command()
def status():
    """Show upload progress and statistics."""
    progress_file = "aws/s3_upload/s3_upload_progress.json"

    if not os.path.exists(progress_file):
        click.echo("❌ No progress file found. Upload hasn't started yet.")
        return

    try:
        with open(progress_file, "r") as f:
            data = json.load(f)

        files = data.get("files", [])
        uploaded = [f for f in files if f.get("uploaded", False)]

        total_files = len(files)
        uploaded_files = len(uploaded)
        progress_pct = (uploaded_files / total_files * 100) if total_files > 0 else 0

        click.echo("📊 DATASET UPLOAD STATUS")
        click.echo("=" * 40)
        click.echo(
            f"Progress: {uploaded_files}/{total_files} files ({progress_pct:.1f}%)"
        )

        if uploaded_files == total_files:
            click.echo("✅ Upload complete!")
        elif uploaded_files > 0:
            click.echo("🔄 Upload in progress")
        else:
            click.echo("⏳ Upload not started")

    except Exception as e:
        click.echo(f"❌ Error reading progress: {e}")


# =============================================================================
# MODEL COMMANDS
# =============================================================================


@models.command()
@click.argument("model_dir")
@click.argument("model_name")
@click.option("--force", is_flag=True, help="Overwrite existing model")
def upload(model_dir: str, model_name: str, force: bool):
    """Upload model weights to S3."""
    if not os.path.exists(model_dir):
        click.echo(f"❌ Model directory not found: {model_dir}")
        sys.exit(1)

    uploader = ModelUploader()

    # Check if model exists
    if not force:
        existing_models = uploader.list_uploaded_models()
        if model_name in existing_models:
            if not click.confirm(f"Model '{model_name}' already exists. Overwrite?"):
                click.echo("Upload cancelled.")
                return

    click.echo(f"Uploading model: {model_name}")
    result = uploader.upload_model(model_dir, model_name)

    click.echo("\n📊 UPLOAD SUMMARY")
    click.echo("=" * 30)
    click.echo(f"Uploaded: {result['uploaded']} files")
    click.echo(f"Skipped:  {result['skipped']} files")
    click.echo(f"Failed:   {result['failed']} files")

    if result["failed"] > 0:
        click.echo("❌ Some files failed to upload")
        sys.exit(1)
    else:
        click.echo("✅ Model upload complete!")


@models.command()
@click.argument("model_name")
@click.argument("download_dir")
def download(model_name: str, download_dir: str):
    """Download model weights from S3."""
    uploader = ModelUploader()

    # Check if model exists
    existing_models = uploader.list_uploaded_models()
    if model_name not in existing_models:
        click.echo(f"❌ Model '{model_name}' not found in S3")
        click.echo("Available models:")
        for model in existing_models:
            click.echo(f"  - {model}")
        sys.exit(1)

    click.echo(f"Downloading model: {model_name}")
    success = uploader.download_model(model_name, download_dir)

    if success:
        click.echo("✅ Model download complete!")
    else:
        click.echo("❌ Model download failed")
        sys.exit(1)


@models.command("list")
def list_models():
    """List all uploaded models."""
    uploader = ModelUploader()
    models = uploader.list_uploaded_models()

    if models:
        click.echo("📦 AVAILABLE MODELS")
        click.echo("=" * 30)
        for model in models:
            click.echo(f"  - {model}")
    else:
        click.echo("No models found in S3")


@models.command()
@click.argument("model_name")
@click.confirmation_option(prompt="Are you sure you want to delete this model?")
def delete(model_name: str):
    """Delete a model from S3."""
    config = get_config()

    try:
        import boto3

        s3_client = boto3.client("s3", region_name=config.aws_region)

        # List and delete all model files
        models_prefix = f"{config.s3_prefix}/models/{model_name}/"
        paginator = s3_client.get_paginator("list_objects_v2")

        files_deleted = 0
        for page in paginator.paginate(Bucket=config.bucket_name, Prefix=models_prefix):
            for obj in page.get("Contents", []):
                s3_client.delete_object(Bucket=config.bucket_name, Key=obj["Key"])
                files_deleted += 1

        click.echo(f"✅ Deleted model '{model_name}' ({files_deleted} files)")

    except Exception as e:
        click.echo(f"❌ Error deleting model: {e}")
        sys.exit(1)


# =============================================================================
# STORAGE COMMANDS
# =============================================================================


@storage.command("list")
@click.option("--prefix", default="", help="S3 prefix to list")
@click.option("--recursive", is_flag=True, help="List recursively")
def list_files(prefix: str, recursive: bool):
    """List files in S3 bucket."""
    config = get_config()

    try:
        import boto3

        s3_client = boto3.client("s3", region_name=config.aws_region)

        full_prefix = f"{config.s3_prefix}/{prefix}".strip("/") + "/"
        if full_prefix == "/":
            full_prefix = ""

        paginator = s3_client.get_paginator("list_objects_v2")

        if not recursive:
            # List only immediate children
            response = s3_client.list_objects_v2(
                Bucket=config.bucket_name, Prefix=full_prefix, Delimiter="/"
            )

            # Show directories
            for common_prefix in response.get("CommonPrefixes", []):
                dir_name = common_prefix["Prefix"].split("/")[-2]
                click.echo(f"📁 {dir_name}/")

            # Show files
            for obj in response.get("Contents", []):
                if obj["Key"] != full_prefix:  # Skip the prefix itself
                    filename = os.path.basename(obj["Key"])
                    size = obj["Size"]
                    click.echo(f"📄 {filename} ({_format_bytes(size)})")
        else:
            # List recursively
            total_files = 0
            total_size = 0

            for page in paginator.paginate(
                Bucket=config.bucket_name, Prefix=full_prefix
            ):
                for obj in page.get("Contents", []):
                    rel_path = obj["Key"][len(full_prefix) :]
                    size = obj["Size"]
                    click.echo(f"📄 {rel_path} ({_format_bytes(size)})")
                    total_files += 1
                    total_size += size

            click.echo(f"\nTotal: {total_files} files, {_format_bytes(total_size)}")

    except Exception as e:
        click.echo(f"❌ Error listing files: {e}")
        sys.exit(1)


@storage.command()
def usage():
    """Show S3 storage usage statistics."""
    config = get_config()

    try:
        import boto3

        s3_client = boto3.client("s3", region_name=config.aws_region)

        # Calculate usage by category
        categories = {
            "dataset1": f"{config.s3_prefix}/dataset1/",
            "dataset2": f"{config.s3_prefix}/dataset2/",
            "dataset3": f"{config.s3_prefix}/dataset3/",
            "models": f"{config.s3_prefix}/models/",
        }

        click.echo("📊 S3 STORAGE USAGE")
        click.echo("=" * 40)

        total_size = 0
        total_files = 0

        for category, prefix in categories.items():
            paginator = s3_client.get_paginator("list_objects_v2")
            cat_size = 0
            cat_files = 0

            for page in paginator.paginate(Bucket=config.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    cat_size += obj["Size"]
                    cat_files += 1

            click.echo(
                f"{category:12} {cat_files:8,} files  {_format_bytes(cat_size):>12}"
            )
            total_size += cat_size
            total_files += cat_files

        click.echo("-" * 40)
        click.echo(
            f"{'TOTAL':12} {total_files:8,} files  {_format_bytes(total_size):>12}"
        )

    except Exception as e:
        click.echo(f"❌ Error calculating usage: {e}")
        sys.exit(1)


@storage.command()
@click.option("--days", default=7, help="Files older than N days")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.confirmation_option(prompt="Are you sure you want to clean up old files?")
def cleanup(days: int, dry_run: bool):
    """Clean up old temporary files."""
    config = get_config()

    try:
        import boto3
        from datetime import datetime, timedelta

        s3_client = boto3.client("s3", region_name=config.aws_region)
        cutoff_date = datetime.now() - timedelta(days=days)

        # Look for temporary files
        temp_prefixes = [
            f"{config.s3_prefix}/tmp/",
            f"{config.s3_prefix}/temp/",
            f"{config.s3_prefix}/.uploads/",
        ]

        files_to_delete = []

        for prefix in temp_prefixes:
            paginator = s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=config.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["LastModified"].replace(tzinfo=None) < cutoff_date:
                        files_to_delete.append(obj["Key"])

        if not files_to_delete:
            click.echo("No old files found to clean up")
            return

        click.echo(f"Found {len(files_to_delete)} old files to clean up")

        if dry_run:
            for key in files_to_delete[:10]:  # Show first 10
                click.echo(f"Would delete: {key}")
            if len(files_to_delete) > 10:
                click.echo(f"... and {len(files_to_delete) - 10} more")
        else:
            # Delete files
            for key in files_to_delete:
                s3_client.delete_object(Bucket=config.bucket_name, Key=key)
            click.echo(f"✅ Deleted {len(files_to_delete)} old files")

    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e}")
        sys.exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _format_bytes(bytes_val: int) -> str:
    """Format bytes as human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


@cli.command()
def setup():
    """Setup AWS credentials and test connection."""
    setup_script = Path(__file__).parent / "s3_upload" / "setup_aws_credentials.sh"
    if setup_script.exists():
        os.system(f"bash {setup_script}")
    else:
        click.echo("❌ Setup script not found")


@cli.command()
@click.option("--config-file", help="Path to custom config file")
def config(config_file: Optional[str]):
    """Show current configuration."""
    if config_file:
        # TODO: Load custom config
        click.echo("Custom config not implemented yet")
    else:
        config = get_config()
        click.echo("📋 CURRENT CONFIGURATION")
        click.echo("=" * 40)
        click.echo(f"S3 Bucket:     {config.bucket_name}")
        click.echo(f"S3 Prefix:     {config.s3_prefix}")
        click.echo(f"AWS Region:    {config.aws_region}")
        click.echo(f"Concurrency:   {config.max_concurrency}")
        click.echo(f"Chunk Size:    {config.chunk_size // (1024 * 1024)} MB")

        if config.max_bandwidth_mbps:
            click.echo(f"Bandwidth:     {config.max_bandwidth_mbps} Mbps")

        click.echo("\nDataset Mappings:")
        for local_dir, dataset_name in config.dataset_mappings.items():
            exists = "✅" if os.path.exists(local_dir) else "❌"
            click.echo(f"  {exists} {dataset_name}: {local_dir}")


if __name__ == "__main__":
    cli()
