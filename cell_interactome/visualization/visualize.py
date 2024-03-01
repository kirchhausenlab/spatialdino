import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from pathlib import Path

plt.style.use("seaborn-v0_8-notebook")


def save_cell_animation(
    img: np.ndarray,
    fname: str,
    visualization_path: Path,
    cmap: str = "gray",
) -> None:
    fig, ax = plt.subplots()
    ax.axis("off")

    artists = []
    for i in range(img.shape[0]):
        artist = ax.imshow(img[i], cmap=cmap, animated=True)
        artists.append([artist])

    ani = ArtistAnimation(fig, artists, interval=200, blit=True, repeat_delay=1000)

    visualization_path.mkdir(parents=True, exist_ok=True)

    ani.save(str(visualization_path.joinpath(fname)), writer="ffmpeg", fps=5, dpi=300)

    plt.close(fig)
