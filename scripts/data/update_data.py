from pathlib import Path
from spatialdino.data.utils import update_pth_file_metadata

if __name__ == "__main__":
    parent_directory = Path("/nfs/scratch2/alavaee/data/processed/llsm")
    update_pth_file_metadata(path_to_search=parent_directory)
