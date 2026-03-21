"""Generate visual quality-check grids from lattice light-sheet experiments.

Walks the data directory to find processed experiment channels, reads a
representative TIF from each, computes the max-intensity Z projection,
and tiles the results into numbered grid images (6x6 by default). Each
grid is saved as a PNG alongside a CSV that maps cell indices to their
source experiment and channel paths, enabling manual visual QC.

Usage::

    python prepare_data_quality_check.py

"""

from pathlib import Path
import skimage.io as io
import numpy as np
from glob import iglob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from spatialdino.data.utils import convert_16bit_to_8bit


def main():
    """Build numbered image grids for visual quality checking.

    Discovers experiment channels under the data directory, reads a single
    TIF per channel, computes a max-intensity projection, and places it into
    a 6x6 grid figure. When the grid is full it is saved as a PNG along with
    a CSV mapping each cell index to its source paths. The process repeats
    for subsequent grids until all channels have been tiled.
    """
    data_path = Path("/nfs/datasync4/AO-LLSM")
    # grid_save_path = Path("./grids")
    grid_save_path = Path("/nfs/scratch2/shared_image_recog_ml/save_grid_dir")

    grid_save_path.mkdir(exist_ok=True, parents=True)

    cs_paths = set()
    curr_id = 0  # index of flattened array in grid (Fortran order)
    grid_count = 0

    GRID_SIZE = (6, 6)
    GRID_AREA = np.prod(GRID_SIZE)

    # Create figure without any padding
    fig = plt.figure(figsize=(18, 26))
    gs = GridSpec(GRID_SIZE[0], GRID_SIZE[1], figure=fig)
    gs.update(wspace=0, hspace=0)  # Set the spacing between axes to 0

    axs = []
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            axs.append(ax)

    columns = ["cs_path", "channel_path", "grid_id", "cell_id"]
    columns_str = ",".join(columns)
    metadata = []
    for cs_path in iglob(str(data_path.joinpath("**/processed/CS*")), recursive=True):
        cs_path = Path(cs_path)

        experiment_path = next(cs_path.glob("Ex*"), None)

        if experiment_path is None or not experiment_path.is_dir():
            continue

        for channel_path in experiment_path.glob("ch*"):
            if curr_id >= GRID_AREA:
                # save the grid
                save_path = grid_save_path.joinpath(str(grid_count))
                save_path.mkdir(exist_ok=True, parents=True)
                fig.savefig(
                    save_path.joinpath("grid.png"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=300,
                )
                with open(save_path.joinpath("metadata.csv"), "w") as f:
                    f.write(columns_str + "\n")
                    for row in metadata:
                        f.write(",".join(row) + "\n")
                for ax in axs:
                    # clear the axes
                    ax.clear()
                    ax.axis("off")
                # clear metadata
                metadata = []
                curr_id = 0
                grid_count += 1
            if channel_path.is_dir():
                channel_name = channel_path.name
                if (cs_path, channel_name) in cs_paths:
                    continue
                cs_paths.add((cs_path, channel_name))
                tif_file = next(channel_path.glob("*.tif"), None)
                if tif_file is not None:
                    array = io.imread(tif_file)
                    array = convert_16bit_to_8bit(array)
                    max_projection = np.max(array, axis=0)
                    ax = axs[curr_id]
                    ax.imshow(max_projection, cmap="gray", aspect="auto")
                    ax.text(
                        max_projection.shape[1] // 2,
                        max_projection.shape[0] // 2,
                        str(curr_id),
                        color="red",
                        fontsize=24,
                        ha="center",
                        va="center",
                        weight="bold",
                    )
                    metadata.append([
                        str(cs_path),
                        channel_path.name,
                        str(grid_count),
                        str(curr_id),
                    ])
                    curr_id += 1
                else:
                    continue


if __name__ == "__main__":
    main()
