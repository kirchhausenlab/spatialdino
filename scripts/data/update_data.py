"""Batch-update metadata in saved ``.pth`` files.

Walks a parent directory of processed lattice light-sheet data and
applies ``update_pth_file_metadata`` to every ``.pth`` file found,
ensuring metadata fields stay consistent with the latest schema.

Usage::

    python update_data.py

"""

from pathlib import Path
from spatialdino.data.utils import update_pth_file_metadata

if __name__ == "__main__":
    parent_directory = Path("/nfs/scratch2/alavaee/data/processed/llsm")
    update_pth_file_metadata(path_to_search=parent_directory)
