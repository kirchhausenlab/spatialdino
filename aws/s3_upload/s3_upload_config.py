"""Configuration for S3 upload operations."""

import os
from dataclasses import dataclass


@dataclass
class S3UploadConfig:
    """Configuration class for S3 upload operations."""

    # AWS S3 Settings
    bucket_name: str = "spatialdino"
    s3_prefix: str = "models"
    aws_region: str = "us-east-1"

    # Upload settings
    chunk_size: int = 100 * 1024 * 1024  # 100MB chunks for multipart
    max_concurrency: int = 4  # Number of parallel uploads
    max_bandwidth_mbps: int | None = None  # No limit by default

    # Retry settings
    max_retries: int = 3
    retry_delay: int = 5  # seconds

    # Progress and logging
    log_level: str = "INFO"
    progress_file: str = "s3_upload_progress.json"

    def __post_init__(self):
        pass


def get_config() -> S3UploadConfig:
    """Get configuration, allowing environment variable overrides."""
    config = S3UploadConfig()

    # Allow environment variable overrides
    config.bucket_name = os.getenv("S3_BUCKET_NAME", config.bucket_name)
    config.s3_prefix = os.getenv("S3_PREFIX", config.s3_prefix)
    config.aws_region = os.getenv("AWS_REGION", config.aws_region)
    config.max_concurrency = int(
        os.getenv("MAX_CONCURRENCY", str(config.max_concurrency))
    )
    config.chunk_size = (
        int(os.getenv("CHUNK_SIZE_MB", str(config.chunk_size // (1024 * 1024))))
        * 1024
        * 1024
    )

    bandwidth_env = os.getenv("MAX_BANDWIDTH_MBPS")
    if bandwidth_env:
        config.max_bandwidth_mbps = int(bandwidth_env)

    return config
