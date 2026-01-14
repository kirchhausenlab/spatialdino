# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Use __file__ instead of __name__ for more reliable path resolution

import functools
import logging
import os
import sys

# So that calling _configure_logger multiple times won't add many handlers
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from .utils import MetricLogger, SmoothedValue

BASE_PATH = Path(__file__).resolve().parents[2]
LOGS_DIR = BASE_PATH / "logs"

# Thread lock for safe logger configuration
_LOGGER_LOCK = threading.Lock()


@functools.lru_cache(maxsize=128)
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        rank: The rank of the process.
    Returns:
        The configured logger.
    """
    with _LOGGER_LOCK:
        logger = logging.getLogger(name)

        # Clear any existing handlers
        if logger.handlers:
            logger.handlers.clear()

        logger.setLevel(level)
        logger.propagate = False

        # Configure formatter
        fmt_prefix = (
            "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
        )
        fmt_message = "%(message)s"
        fmt = fmt_prefix + fmt_message
        datefmt = "%Y%m%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        # Configure stdout handler for main worker
        if rank == 0:
            try:
                handler = logging.StreamHandler(stream=sys.stdout)
                handler.setLevel(logging.DEBUG)
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            except Exception as e:
                logger.error(f"Failed to set up stdout logging: {e}")

        # Configure file handler for all workers
        if output:
            try:
                # Convert output to Path object for consistent handling
                output_path = Path(output)

                # Determine log file path
                if output_path.suffix in (".txt", ".log"):
                    filename = output_path
                else:
                    filename = output_path / "logs" / "log.txt"

                # Add rank suffix for distributed training
                if rank != 0:
                    filename = filename.with_suffix(f".rank{rank}{filename.suffix}")

                # Ensure directory exists
                filename.parent.mkdir(parents=True, exist_ok=True)

                # Set up rotating file handler with proper permissions
                handler = RotatingFileHandler(
                    filename,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5,
                    mode="a",
                    encoding="utf-8",
                    delay=True,  # Don't open file until first log
                )

                # Set file permissions (rw-r--r--)
                os.chmod(filename, 0o644)

                handler.setLevel(logging.DEBUG)
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            except Exception as e:
                logger.error(f"Failed to set up file logging: {e}")
                # Continue with console logging if file logging fails

        return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
    rank: int = 0,
) -> None:
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        capture_warnings: Whether warnings should be captured as logs.
        rank: The rank of the process.
    """
    try:
        logging.captureWarnings(capture_warnings)
        _configure_logger(name, level=level, output=output, rank=rank)
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
        logging.error(f"Failed to setup custom logging: {e}")