"""AWS S3 Upload Module for Cell Interactome."""

__version__ = "1.0.0"

from .s3_upload_config import get_config
from .model_uploader import ModelUploader

__all__ = ["get_config", "ModelUploader"]
