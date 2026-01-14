from typing import Any
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from pathlib import Path
import tempfile
from spatialdino.data.transforms import convert_16bit_to_8bit
from spatialdino.data.dataset import (
    ZFlattenMode,
    flatten_z_axis,
    image_to_color_mode,
)
from spatialdino.data.utils import reconstruct_3d_image_from_voxels
from threading import Thread
from loguru import logger
from tqdm.auto import tqdm
from spatialdino.visualization import REPORTS_DIR


def save_2d_image_from_voxels(
    stack_dir: Path,
    save_path: Path,
    z_flatten_mode: ZFlattenMode,
    target_height: int = 224,
    target_width: int = 224,
) -> None:
    """_summary_
    Store the image as a .png

    Parameters
    ----------
    stack_dir : Path
        _description_
    save_path : Path
        _description_
    z_flatten_mode : ZFlattenMode
        _description_
    target_height : int, optional
        _description_, by default 224
    target_width : int, optional
        _description_, by default 224
    """ """Store 2d image as .png"""
    image_3d = reconstruct_3d_image_from_voxels(stack_dir=stack_dir)
    image_2d = flatten_z_axis(image=image_3d, mode=z_flatten_mode)
    image_2d = image_to_color_mode(image=image_2d)
    image_2d = image_2d.transpose(1, 2, 0)
    image_2d = convert_16bit_to_8bit(image_2d)
    im = Image.fromarray(image_2d, mode="RGB")
    im.thumbnail((target_height, target_width), resample=Image.Resampling.LANCZOS)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(save_path)


def save_2d_imgs(img: np.ndarray, save_path: Path) -> None:
    """Store image as .jpeg, .png

    Parameters
    ----------
    img: np.ndarray
        Image to be stored, should be 8-bit image

    save_path: Path
        Path to store the image
    """
    img = img.transpose(1, 2, 0)
    im = Image.fromarray(img)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(save_path)


def save_2d_video_from_voxels(
    experiment_dir: Path,
    save_path: Path,
    z_flatten_mode: ZFlattenMode,
    **kwargs: Any,
) -> None:
    """Store 2d video as .mp4

    Parameters
    ----------
    experiment_dir: Path
        Path to the directory containing the experiment with the specific color channel (e.g. 20230407_p5_p55_sCMOS_Zhenyu_rVSVGpHrodo_iNeuronsprocessedCS3_Rabies_G_rVSV_1to10_5min_Incubation__Ex09_488_30mW_560_100mW_642_50mW_z0p5/488nm_CamA)

    save_path: Path
        Path to store the image

    **kwargs: Any
        Additional arguments to be passed to plt.subplots
    """
    threads = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        for stack_dir in experiment_dir.iterdir():
            thread = Thread(
                target=save_2d_image_from_voxels,
                args=(
                    stack_dir,
                    tmp_dir.joinpath(f"{stack_dir.name}.png"),
                    z_flatten_mode,
                    kwargs["target_height"],
                    kwargs["target_width"],
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in tqdm(threads, desc="Creating images"):
            thread.join()

        artists = []

        logger.info("Creating animation in %s" % save_path)
        image_paths = list(tmp_dir.iterdir())
        image_paths = sorted(
            image_paths, key=lambda path: int(path.stem.split("_")[-1])
        )
        with Image.open(image_paths[0]) as first_image:
            first_image = first_image.convert("RGB")
            width, height = first_image.size

        dpi = 300
        fig, ax = plt.subplots(
            figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False, **kwargs
        )
        ax.set_position((0, 0, 1, 1))
        ax.axis("off")

        for image_path in tqdm(
            image_paths, desc="Creating animation", total=len(image_paths)
        ):
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                artist = ax.imshow(img, animated=True, aspect="auto")
                artists.append([artist])

        ani = ArtistAnimation(fig, artists, interval=200, blit=True, repeat_delay=1000)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(save_path, writer="ffmpeg", fps=5, dpi=dpi)

        plt.close("all")


def movie_for_viewing_all_inference_images(
    input_path: Path = REPORTS_DIR.joinpath("2d_preds"),
    output_path: Path = REPORTS_DIR.joinpath("2d_preds", "video.mp4"),
    fps: int = 1,
):
    """
    input_path: Path
        Path to the directory containing the images
    output_path: Path
        Path to store the video
    fps: int
        Frames per second
    """
    image_files = sorted(list(input_path.glob("*.png")))

    if not image_files:
        print("No PNG images found in the input directory.")
        return

    # Read the first image to get dimensions
    with Image.open(image_files[0]) as img:
        img = img.convert("RGB")
        img_array = np.array(img)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.axis("off")

    # Create a list to store all frames
    frames = []

    # Load all images and create frames
    for image_file in tqdm(image_files, desc="Loading images"):
        with Image.open(image_file) as img:
            img = img.convert("RGB")
            img_array = np.array(img)
            frame = [ax.imshow(img_array, animated=True)]
            frames.append(frame)

    # Create the animation
    ani = ArtistAnimation(
        fig, frames, interval=1000 / fps, blit=True, repeat_delay=1000
    )

    # Save the animation
    ani.save(output_path, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Video created successfully: {output_path}")
