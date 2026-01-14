import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from scipy.optimize import linear_sum_assignment
from skimage import io


def generate_dict(
    file_path: Path,
    exp_root: str,
    FRAMES: List[int] = [],
) -> Tuple[
    Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]
]:
    seg, pca, raw = {}, {}, {}
    for folder_name in file_path.iterdir():
        if str(folder_name.stem).split("_")[:3] == exp_root.split("_")[:3]:
            frame = str(folder_name.stem).split("_")[3]
            convert_to_list = list(frame)
            frame_num = convert_to_list[-3:]
            frame_num = int("".join(frame_num))
            FRAMES.append(frame_num)
            for tif_file in folder_name.glob("*.tif"):
                if ".tif" in tif_file.suffix:
                    if "in_seg" in tif_file.stem:
                        seg[frame_num] = tif.imread(tif_file.parent / "in_seg.tif")
                    if "volume" in tif_file.stem:
                        raw[frame_num] = tif.imread(tif_file.parent / "volume.tif")
                    if "pca_features" in tif_file.stem:
                        # doing this to explicitly load each type of pca feature
                        pca[frame_num] = tif.imread(
                            tif_file.parent / "pca_features.tif"
                        )
                    if "density_map" in tif_file.stem:
                        pca[frame_num] = tif.imread(tif_file.parent / "density_map.tif")
    FRAMES.sort()
    return seg, pca, raw, FRAMES


def plot_fig(
    seg: Dict[int, np.ndarray],
    FRAMES: List[int],
    plot_figs: bool = False,
    z_diff: int = 5,
    cols: int = 12,
) -> None:
    if plot_figs:
        num_frames = seg[FRAMES[0]].shape[0]
        print(
            f"We have {num_frames} frames for {FRAMES[0]} of shape {seg[FRAMES[0]].shape}"
        )

        z_diff = 5
        figs_to_show = math.ceil(num_frames / z_diff)
        print(f"We will show {figs_to_show} frames")

        cols = 6 * 2
        rows = math.ceil(figs_to_show / (cols // 2))
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        axs = axs.flatten()

        for i in range(figs_to_show):
            frame_idx = i * z_diff
            if frame_idx < num_frames:
                ax_idx = i
                axs[ax_idx].imshow(seg[FRAMES[0]][frame_idx], cmap="tab20")
                axs[ax_idx].set_title(f"Frame {frame_idx}")
                axs[ax_idx].axis("off")

        for i in range(figs_to_show, rows * cols):
            axs[i].axis("off")
        plt.tight_layout()
        plt.show()


def calculate_centroids(
    segmentation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    all_labels = np.unique(segmentation)
    centroids = np.zeros((len(all_labels), 3))
    centroids_dict = {}
    for idx, label in enumerate(all_labels):
        if label == 0:
            continue
        mask = segmentation == label
        centroid = np.mean(np.where(mask), axis=1)
        centroids_dict[idx] = centroid
        centroids[idx] = centroid
    return centroids, all_labels, centroids_dict


def enerate_cost_matrix(centroids1: np.ndarray, centroids2: np.ndarray) -> np.ndarray:
    cost_matrix = np.zeros((len(centroids1), len(centroids2)))
    for i, centroid1 in enumerate(centroids1):
        for j, centroid2 in enumerate(centroids2):
            cost_matrix[i, j] = np.linalg.norm(centroid1 - centroid2)
    return cost_matrix


def hungarian_algorithm(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def match_labels(
    labels_frame1: np.ndarray,
    labels_frame2: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    seg: Dict[int, np.ndarray],
    frame1: int,
    frame2: int,
) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    label_mapping = {}
    max_labels = max(labels_frame1.max(), labels_frame2.max())
    unmatched_label_value = max_labels + 1
    for i, j in zip(row_ind, col_ind):
        if i >= len(labels_frame1) or j >= len(labels_frame2):
            # print(f"Skipping out of bounds indices: i={i}, j={j}")
            continue
        if labels_frame1[i] == 0 or labels_frame2[j] == 0:
            continue
        label_mapping[labels_frame2[j]] = labels_frame1[i]
    # copy the frame2 segmentation and update the labels
    updated_seg_frame2 = np.zeros_like(seg[frame2])
    for label2 in labels_frame2:
        if label2 == 0:  # ignore background
            continue
        label_mask = seg[frame2] == label2
        if label2 in label_mapping:
            updated_seg_frame2[label_mask] = label_mapping[label2]
        else:
            updated_seg_frame2[label_mask] = unmatched_label_value
            unmatched_label_value += 1
    # diff in the maps
    diff_map = updated_seg_frame2 - seg[frame1]
    return updated_seg_frame2, label_mapping, diff_map


@lru_cache()
def get_unique_labels_and_pca(
    segmentation: np.ndarray,
    pca_features: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    all_labels = np.unique(segmentation)
    pca_feats = np.zeros((len(all_labels), pca_features.shape[-1]))
    pca_feats_dict = {}
    for i, label in enumerate(all_labels):
        if label == 0:
            continue
        mask = segmentation == label
        pca1_label = pca_features[mask]
        pca_feat = np.mean(pca1_label, axis=0)
        # print(f"label {label} has shape {pca1_label.shape} detections, mean is {np.mean(pca1_label, axis=0)}")
        pca_feats_dict[label] = pca_feat
        pca_feats[i] = pca_feat
    return pca_feats, pca_feats_dict


def stack_frames_across_time(
    input_data: List[np.ndarray],
) -> np.ndarray:
    stacked_data = np.stack(input_data, axis=0)
    return stacked_data


def save_tiffs(
    curr_path: Path,
    raw_old: np.ndarray,
    raw_updated: np.ndarray,
    experiment_type: str,
    experiment_num: str,
    updated_seg: np.ndarray,
    old_seg: np.ndarray,
    diff_map: np.ndarray,
    save_time_movie: bool = False,
) -> None:
    save_path = curr_path.joinpath(experiment_type)
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = save_path.joinpath(experiment_num)
    save_path.mkdir(exist_ok=True, parents=True)
    io.imsave(save_path.joinpath("updated_seg.tif"), updated_seg)
    io.imsave(save_path.joinpath("old_seg.tif"), old_seg)
    io.imsave(save_path.joinpath("diff_map.tif"), diff_map.astype(np.uint8) * 255)

    if save_time_movie:
        save_path_time = save_path.joinpath("time_tracking")
        save_path_time.mkdir(exist_ok=True, parents=True)
        stacked_seg = stack_frames_across_time([old_seg, updated_seg])
        io.imsave(save_path_time.joinpath("stacked_seg.tif"), stacked_seg)
        stacked_raw = stack_frames_across_time([raw_old, raw_updated])
        io.imsave(save_path_time.joinpath("stacked_raw.tif"), stacked_raw)
