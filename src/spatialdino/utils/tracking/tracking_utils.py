from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from natsort import natsorted
from skimage import io
from trackpy.predict import NearestVelocityPredict
from tqdm import tqdm
from collections import Counter, defaultdict
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity

from spatialdino.tracking.simple_decay_correction import (
    apply_decay_correction_to_intensities,
    get_decay_correction_for_dino_track,
)
from spatialdino.utils.tracking.plotting import plot_track_lengths
from matplotlib.backends.backend_pdf import PdfPages


def compute_cosine_distances(
    reference_features: np.ndarray, candidate_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine distances between reference features and candidate particles.

    Args:
        reference_features: 1D array of shape (n_features,) - reference particle features
        candidate_features: 2D array of shape (n_particles, n_features) - candidate particles features

    Returns:
        Tuple of:
        - overall_distances: 1D array of shape (n_particles,) - overall cosine distance for each particle
        - per_feature_distances: 2D array of shape (n_particles, n_features) - per-feature cosine distances
    """
    # Overall cosine distances using all features
    reference_reshaped = reference_features.reshape(1, -1)
    cosine_similarities = cosine_similarity(reference_reshaped, candidate_features)[0]
    overall_distances = 1 - cosine_similarities

    # Per-feature cosine distances
    per_feature_distances = np.zeros((
        candidate_features.shape[0],
        candidate_features.shape[1],
    ))

    for feat_idx in range(candidate_features.shape[1]):
        ref_feat = reference_features[feat_idx : feat_idx + 1].reshape(1, -1)
        candidate_feats = candidate_features[:, feat_idx : feat_idx + 1]

        # Handle zero vectors
        ref_norm = np.linalg.norm(ref_feat)
        candidate_norms = np.linalg.norm(candidate_feats, axis=1)

        if ref_norm == 0:
            # If reference is zero, distance is 1 for non-zero candidates, 0 for zero candidates
            feat_distances = np.where(candidate_norms == 0, 0, 1)
        else:
            # Calculate cosine similarities
            similarities = np.zeros(len(candidate_norms))
            for i, (candidate_feat, candidate_norm) in enumerate(
                zip(candidate_feats, candidate_norms)
            ):
                if candidate_norm == 0:
                    similarities[i] = (
                        0  # Zero vector has 0 similarity with non-zero vector
                    )
                else:
                    similarities[i] = np.dot(ref_feat[0], candidate_feat) / (
                        ref_norm * candidate_norm
                    )
            feat_distances = 1 - similarities

        per_feature_distances[:, feat_idx] = feat_distances

    return overall_distances, per_feature_distances


def get_file_paths(base_dir: Path) -> List[Path]:
    experiments = natsorted(list(base_dir.iterdir()))
    lst = []
    for f in experiments:
        if f.is_dir():
            has_matching_file = any(
                "raw_segmentation_mask.tif" in file.name
                or "instance_seg.tif" in file.name
                for file in f.iterdir()
            )
            if has_matching_file:
                lst.append(f.stem)
    file_paths = [base_dir.joinpath(f) for f in lst]
    assert all([p.is_dir() for p in file_paths]), "Not all paths are directories"
    return file_paths


def filter_particles_at_t(
    track_features: pd.DataFrame,
    t: int,
    x_val: float,
    y_val: float,
    range_value: float = 10.0,
) -> pd.DataFrame:
    x_range = [x_val - range_value, x_val + range_value]
    y_range = [y_val - range_value, y_val + range_value]
    print(
        f"Filtering particles at t={t} within x_range={x_range} and y_range={y_range}"
    )
    return track_features[
        (track_features["t"] == t)
        & (track_features["x"] >= x_range[0])
        & (track_features["x"] <= x_range[1])
        & (track_features["y"] >= y_range[0])
        & (track_features["y"] <= y_range[1])
    ]


def load_centroids(
    file_paths: List[Path],
    centroid_type: Literal["centroids", "centroids_new"] = "centroids",
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], int, int]:
    centroids, detections = {}, {}
    min_detections = np.inf
    max_detections = 0
    for i, file_path in enumerate(file_paths):
        if not file_path.joinpath(f"{centroid_type}.tif").exists():
            continue
        centroid = io.imread(file_path.joinpath(f"{centroid_type}.tif"))
        centroids[i] = centroid
        detections[i] = centroid.shape[0]
        min_detections = min(min_detections, detections[i])
        max_detections = max(max_detections, detections[i])

    return centroids, detections, int(min_detections), int(max_detections)


def cme_fixes(
    df_cme: pd.DataFrame,
    df_dino: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_cme["t"] = df_cme["t_starting"] + df_cme["t"] - 1
    df_cme["z"] = 168 - df_cme["z"]
    df_dino["t"] = df_dino["t"] + 1
    df_dino["t_starting"] = df_dino["t_starting"] + 1
    return df_cme, df_dino


def summarize_tracks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        df.groupby("ID")
        .agg({
            "track_length": "first",
            "t": ["min", "max"],
            "x": ["min", "max"],
            "y": ["min", "max"],
            "z": ["min", "max"],
        })
        .round(2)
    )
    summary.columns = [
        "track_length",
        "t_min",
        "t_max",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    ]
    summary = summary.sort_values("track_length", ascending=False).reset_index()
    return summary


def sample_centroids(centroids: Dict[int, np.ndarray], sample_size: int) -> np.ndarray:
    sampled_pos = []
    for i in range(len(centroids)):
        if centroids[i] is None:
            continue
        frame_centroids = centroids[i]
        if frame_centroids.shape[0] > sample_size:
            indices = np.random.choice(
                frame_centroids.shape[0], sample_size, replace=False
            )
            sampled_pos.append(frame_centroids[indices])
        else:
            sampled_pos.append(frame_centroids)
    return np.vstack(sampled_pos)


def create_velocity_predictor(positions: np.ndarray) -> NearestVelocityPredict:
    initial_guess_vels = np.zeros_like(positions)
    return NearestVelocityPredict(
        initial_guess_positions=positions,
        initial_guess_vels=initial_guess_vels,
        pos_columns=["z", "y", "x"],
    )


def postprocess_tracking(linked: pd.DataFrame) -> pd.DataFrame:
    linked.sort_values(by=["particle", "t"], inplace=True)
    linked.reset_index(drop=True, inplace=True)
    linked["t_0"] = linked.groupby("particle")["t"].transform("first")
    linked["track_length"] = linked.groupby("particle")["particle"].transform("count")
    linked["t_end"] = linked.groupby("particle")["t"].transform("last")
    linked["track_id"] = linked["particle"].copy()

    return linked


def filter_tracks(linked: pd.DataFrame, min_length: int = 10) -> pd.DataFrame:
    linked_refined = tp.filter_stubs(linked.rename(columns={"t": "frame"}), min_length)
    linked = linked_refined.rename(columns={"frame": "t"})
    return linked


def benchmark_tracks(
    linked1: pd.DataFrame,
    linked2: pd.DataFrame,
    track_length_1: Optional[int] = None,
    track_length_2: Optional[int] = None,
    title: str = "Feature Evolution Comparison",
    plot_1_name: str = "SNR 7",
    plot_2_name: str = "SNR 4",
    r: int = 2,
    c: int = 3,
    figsize: Tuple[int, int] = (15, 8),
) -> None:
    id_to_use = "ID" if "ID" in linked1.columns else "track_id"
    # Handle track_length_1
    max_length_1 = linked1["track_length"].max()
    if track_length_1 is None:
        track_length_1 = max_length_1
    elif track_length_1 > max_length_1:
        print(
            f"Warning: Requested track_length_1={track_length_1} exceeds maximum available length {max_length_1}. Using {max_length_1} instead."
        )
        track_length_1 = max_length_1

    # Filter for tracks with the desired length
    longest_track_1 = linked1[linked1["track_length"] == track_length_1].copy()

    # Check if we found any tracks
    if len(longest_track_1) == 0:
        print(
            f"Available track lengths in linked1: {sorted(linked1['track_length'].unique())}"
        )
        raise ValueError(f"No tracks found with length {track_length_1} in linked1")

    track_id_1 = longest_track_1[id_to_use].iloc[0]
    print(
        f"Longest track in linked1 {plot_1_name}: track_id={track_id_1}, length={longest_track_1['track_length'].iloc[0]}"
    )

    # Handle track_length_2
    max_length_2 = linked2["track_length"].max()
    if track_length_2 is None:
        track_length_2 = max_length_2
    elif track_length_2 > max_length_2:
        print(
            f"Warning: Requested track_length_2={track_length_2} exceeds maximum available length {max_length_2}. Using {max_length_2} instead."
        )
        track_length_2 = max_length_2

    # Filter for tracks with the desired length
    longest_track_2 = linked2[linked2["track_length"] == track_length_2].copy()

    # Check if we found any tracks
    if len(longest_track_2) == 0:
        print(
            f"Available track lengths in linked2: {sorted(linked2['track_length'].unique())}"
        )
        raise ValueError(f"No tracks found with length {track_length_2} in linked2")

    track_id_2 = longest_track_2[id_to_use].iloc[0]
    print(
        f"Longest track in linked2 {plot_2_name}: track_id={track_id_2}, length={longest_track_2['track_length'].iloc[0]}"
    )

    fig, axes = plt.subplots(r, c, figsize=figsize)
    fig.suptitle(f"{title}: Longest Tracks {plot_1_name} vs {plot_2_name}", fontsize=16)

    # Plot features for longest track in linked1 (SNR 7)
    for i in range(3):
        axes[0, i].plot(
            longest_track_1["t"],
            longest_track_1[f"feature_{i}"],
            "o-",
            linewidth=2,
            markersize=4,
            label=f"{plot_1_name} (track {track_id_1})",
        )
        axes[0, i].set_title(f"Feature {i} - {plot_1_name}")
        axes[0, i].set_xlabel("Time")
        axes[0, i].set_ylabel(f"Feature {i}")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()

    # Plot features for longest track in linked2 (SNR 4)
    for i in range(3):
        axes[1, i].plot(
            longest_track_2["t"],
            longest_track_2[f"feature_{i}"],
            "o-",
            linewidth=2,
            markersize=4,
            color="orange",
            label=f"{plot_2_name} (track {track_id_2})",
        )
        axes[1, i].set_title(f"Feature {i} - {plot_2_name}")
        axes[1, i].set_xlabel("Time")
        axes[1, i].set_ylabel(f"Feature {i}")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()

    plt.tight_layout()
    plt.show()

    # Create overlay plots for direct comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Direct Feature Comparison: Longest Tracks {plot_1_name} vs {plot_2_name}",
        fontsize=16,
    )

    for i in range(3):
        axes[i].plot(
            longest_track_1["t"],
            longest_track_1[f"feature_{i}"],
            "o-",
            linewidth=2,
            markersize=4,
            label=f"{plot_1_name} (track {track_id_1})",
        )
        axes[i].plot(
            longest_track_2["t"],
            longest_track_2[f"feature_{i}"],
            "o-",
            linewidth=2,
            markersize=4,
            alpha=0.7,
            label=f"{plot_2_name} (track {track_id_2})",
        )
        axes[i].set_title(f"Feature {i} Comparison")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(f"Feature {i}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    # Create overlay plots with intensity and all features in single graphs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Intensity and Features Overlay: Raw Values - {plot_1_name} vs {plot_2_name}",
        fontsize=16,
    )

    # SNR 7 - Raw values with dual y-axis
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    # Plot intensity on left y-axis
    line1 = ax1.plot(
        longest_track_1["t"],
        longest_track_1["intensity_raw"],
        "o-",
        linewidth=3,
        markersize=4,
        color="cyan",
        label="Intensity Raw",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Intensity Raw", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Plot features on right y-axis
    colors = ["red", "green", "blue"]
    lines2 = []
    for i in range(3):
        line = ax1_twin.plot(
            longest_track_1["t"],
            longest_track_1[f"feature_{i}"],
            "o-",
            linewidth=2,
            markersize=3,
            color=colors[i],
            label=f"Feature {i}",
        )
        lines2.extend(line)

    ax1_twin.set_ylabel("Feature Values", color="red")
    ax1_twin.tick_params(axis="y", labelcolor="red")
    ax1.set_title(f"{plot_1_name} (track {track_id_1}) - Raw Values")
    ax1.grid(True, alpha=0.3)

    # Combined legend
    all_lines = line1 + lines2
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc="upper left")

    # SNR 4 - Raw values with dual y-axis
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    # Plot intensity on left y-axis
    line3 = ax2.plot(
        longest_track_2["t"],
        longest_track_2["intensity_raw"],
        "o-",
        linewidth=3,
        markersize=4,
        color="black",
        label="Intensity Raw",
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Intensity Raw", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Plot features on right y-axis
    lines4 = []
    for i in range(3):
        line = ax2_twin.plot(
            longest_track_2["t"],
            longest_track_2[f"feature_{i}"],
            "o-",
            linewidth=2,
            markersize=3,
            color=colors[i],
            label=f"Feature {i}",
        )
        lines4.extend(line)

    ax2_twin.set_ylabel("Feature Values", color="red")
    ax2_twin.tick_params(axis="y", labelcolor="red")
    ax2.set_title(f"{plot_2_name} (track {track_id_2}) - Raw Values")
    ax2.grid(True, alpha=0.3)

    # Combined legend
    all_lines2 = line3 + lines4
    all_labels2 = [l.get_label() for l in all_lines2]
    ax2.legend(all_lines2, all_labels2, loc="upper left")

    plt.tight_layout()
    plt.show()

    # Create intensity ratio plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Feature to Intensity Ratios: {plot_1_name} vs {plot_2_name}", fontsize=16
    )

    # Calculate ratios for SNR 7
    ratios_1 = {}
    for i in range(3):
        ratios_1[f"feature_{i}_ratio"] = (
            longest_track_1[f"feature_{i}"] / longest_track_1["intensity_raw"]
        )

    # Calculate ratios for SNR 4
    ratios_2 = {}
    for i in range(3):
        ratios_2[f"feature_{i}_ratio"] = (
            longest_track_2[f"feature_{i}"] / longest_track_2["intensity_raw"]
        )

    # Plot all feature ratios together for SNR 7
    axes[0].plot(
        longest_track_1["t"],
        ratios_1["feature_0_ratio"],
        "o-",
        linewidth=2,
        markersize=3,
        label="Feature 0/Intensity",
    )
    axes[0].plot(
        longest_track_1["t"],
        ratios_1["feature_1_ratio"],
        "o-",
        linewidth=2,
        markersize=3,
        label="Feature 1/Intensity",
    )
    axes[0].plot(
        longest_track_1["t"],
        ratios_1["feature_2_ratio"],
        "o-",
        linewidth=2,
        markersize=3,
        label="Feature 2/Intensity",
    )
    axes[0].set_title(
        f"All Feature/Intensity Ratios - {plot_1_name} (track {track_id_1})"
    )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Feature/Intensity Ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot all feature ratios together for SNR 4
    axes[1].plot(
        longest_track_2["t"],
        ratios_2["feature_0_ratio"],
        "o-",
        linewidth=2,
        markersize=3,
        label="Feature 0/Intensity",
    )
    axes[1].plot(
        longest_track_2["t"],
        ratios_2["feature_1_ratio"],
        "o-",
        linewidth=2,
        markersize=3,
        label="Feature 1/Intensity",
    )
    axes[1].plot(
        longest_track_2["t"],
        ratios_2["feature_2_ratio"],
        "o-",
        linewidth=2,
        markersize=3,
        label="Feature 2/Intensity",
    )
    axes[1].set_title(
        f"All Feature/Intensity Ratios - {plot_2_name} (track {track_id_2})"
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Feature/Intensity Ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def test_search_range(
    pred: NearestVelocityPredict,
    df: pd.DataFrame,
    adaptive_stops: List[float] = [1.0, 2.0, 3.0, 4.0],
    search_range: float = 35.0,
    memory: int = 100,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    linked_dfs = []
    summary_dfs = []
    adaptive_step: float = 0.95
    centroid_cols: List[str] = ["z", "y", "x"]
    feature_cols: List[str] = ["feature_0", "feature_1", "feature_2"]
    print(f"Testing {len(adaptive_stops)} adaptive stops")
    fig, ax = plt.subplots(1, len(adaptive_stops), figsize=(20, 5))
    for i, adaptive_stop in enumerate(adaptive_stops):
        linked = pred.link_df(
            df,
            search_range=search_range,
            adaptive_stop=adaptive_stop,
            adaptive_step=adaptive_step,
            memory=memory,
            pos_columns=centroid_cols + feature_cols,
            t_column="t",
            predictor=pred,
        )
        linked = postprocess_tracking(linked)
        summary_df = plot_track_lengths(linked, ax=ax[i], experiment_name="test")  # type: ignore
        summary_dfs.append(summary_df)
        linked_dfs.append(linked)
    return linked_dfs, summary_dfs


def test_memory(
    pred: NearestVelocityPredict,
    df: pd.DataFrame,
    adaptive_stop: float = 1.0,
    adaptive_step: float = 0.95,
    search_range: float = 35.0,
    memory_lst: List[int] = [5, 10, 20, 50],
) -> List[pd.DataFrame]:
    linked_dfs = []
    centroid_cols: List[str] = ["z", "y", "x"]
    feature_cols: List[str] = ["feature_0", "feature_1", "feature_2"]
    fig, ax = plt.subplots(1, len(memory_lst), figsize=(20, 5))
    print(f"Testing {len(memory_lst)} memories")
    for i, memory in enumerate(memory_lst):
        linked = pred.link_df(
            df,
            search_range=search_range,
            adaptive_stop=adaptive_stop,
            adaptive_step=adaptive_step,
            memory=memory,
            pos_columns=centroid_cols + feature_cols,
            t_column="t",
            predictor=pred,
        )
        linked = postprocess_tracking(linked)
        linked["track_length"].groupby(linked["track_id"]).first().plot.hist(
            bins=20, edgecolor="black", ax=ax[i]
        )
        ax[i].set_title(f"memory = {memory}")
        for bar in ax[i].patches:
            height = bar.get_height()
            if height > 0:
                x_center = bar.get_x() + bar.get_width() / 2
                ax[i].text(
                    x_center,  # x-coordinate
                    height,  # y-coordinate
                    f"{int(height)}",  # label
                    ha="center",  # horizontal alignment
                    va="bottom",  # vertical alignment
                    fontsize=8,
                    rotation=0,
                )
        linked_dfs.append(linked)
    plt.show()
    return linked_dfs


def adaptive_distance(
    x: np.ndarray,
    y: np.ndarray,
    max_spatial_dist: float = 10.0,
    min_feature_weight: float = 0.1,
    max_feature_weight: float = 0.8,
    z_scale_factor: float = 1.0,
) -> float:
    # Apply z-scaling for isotropic distance calculation
    x_scaled = x[:3].copy()
    y_scaled = y[:3].copy()
    x_scaled[2] *= z_scale_factor
    y_scaled[2] *= z_scale_factor

    spatial_diff = x_scaled - y_scaled
    feature_diff = x[3:] - y[3:]

    spatial_dist = np.sqrt(np.sum(spatial_diff**2))
    feature_dist = np.sqrt(np.sum(feature_diff**2))

    # Adaptive weighting: closer particles rely more on features
    spatial_ratio = min(spatial_dist / max_spatial_dist, 1.0)
    feature_weight = min_feature_weight + (max_feature_weight - min_feature_weight) * (
        1 - spatial_ratio
    )

    return spatial_dist + feature_weight * feature_dist


def weighted_euclidean_distance(
    x: np.ndarray,
    y: np.ndarray,
    spatial_weight: float = 1.0,
    feature_weight: float = 0.3,
    z_scale_factor: float = 1.0,
) -> float:
    # Apply z-scaling for isotropic distance calculation
    x_scaled = x[:3].copy()
    y_scaled = y[:3].copy()
    x_scaled[2] *= z_scale_factor
    y_scaled[2] *= z_scale_factor

    spatial_diff = x_scaled - y_scaled
    feature_diff = x[3:] - y[3:]

    spatial_dist_sq = np.sum(spatial_diff**2) * spatial_weight**2
    feature_dist_sq = np.sum(feature_diff**2) * feature_weight**2

    return np.sqrt(spatial_dist_sq + feature_dist_sq)


def find_proximate_tracks(
    ref_track_id: int,
    ref_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    proximity_threshold: float = 3.0,
    z_scale_factor: float = 1.0,
) -> Tuple[Dict[int, List[int]], set, List[Dict]]:
    ref_track = ref_df[ref_df["ID"] == ref_track_id].copy()
    if len(ref_track) == 0:
        return {}, set(), []

    ref_track = ref_track.sort_values("t")
    proximate_by_time = {}
    all_proximate_ids = set()
    correspondences = []

    for _, ref_point in ref_track.iterrows():
        t = ref_point["t"]
        # Apply z-scaling for isotropic distance calculation
        ref_pos = np.array([
            ref_point["x"],
            ref_point["y"],
            ref_point["z"] * z_scale_factor,
        ])

        # Get all tracks at this time point - more efficient than iterating
        comp_at_t = comp_df[comp_df["t"] == t].copy()

        if len(comp_at_t) == 0:
            proximate_by_time[t] = []
            continue

        # Vectorized distance calculation for efficiency with z-scaling
        comp_positions = comp_at_t[["x", "y", "z"]].values
        comp_positions[:, 2] *= z_scale_factor  # Scale z-coordinates
        distances = np.linalg.norm(comp_positions - ref_pos, axis=1)

        # Find indices within threshold
        proximate_indices = np.where(distances <= proximity_threshold)[0]
        proximate_ids = comp_at_t.iloc[proximate_indices]["ID"].tolist()

        proximate_by_time[t] = proximate_ids
        all_proximate_ids.update(proximate_ids)

        # Store detailed correspondence data
        for idx in proximate_indices:
            comp_point = comp_at_t.iloc[idx]
            correspondences.append({
                "ref_track_id": ref_track_id,
                "ref_track_length": ref_point["track_length"],
                "comp_track_id": comp_point["ID"],
                "comp_track_length": comp_point["track_length"],
                "t": t,
                "distance": distances[idx],
                "ref_x": ref_point["x"],
                "ref_y": ref_point["y"],
                "ref_z": ref_point["z"],
                "comp_x": comp_point["x"],
                "comp_y": comp_point["y"],
                "comp_z": comp_point["z"],
            })

    return proximate_by_time, all_proximate_ids, correspondences


def create_correspondence_table(
    correspondence_data: Dict[str, pd.DataFrame], reference_label: str
) -> Dict[str, pd.DataFrame]:
    """
    Create simple correspondence tables showing which tracks match between datasets.

    Returns:
        Dictionary mapping comparison dataset names to correspondence tables
    """
    correspondence_tables = {}

    for comp_label, corr_df in correspondence_data.items():
        if len(corr_df) == 0:
            correspondence_tables[comp_label] = pd.DataFrame(
                columns=[
                    f"{reference_label}_ID",
                    f"{reference_label}_Length",
                    f"{comp_label}_ID",
                    f"{comp_label}_Length",
                    "Distance",
                ]
            )
            continue

        # Get the closest match for each reference track
        # (track with minimum average distance across all time points)
        closest_matches = []

        for ref_id in corr_df["ref_track_id"].unique():
            ref_corr = corr_df[corr_df["ref_track_id"] == ref_id].copy()

            # Get average distance for each comparison track
            avg_distances = ref_corr.groupby("comp_track_id")["distance"].mean()
            closest_comp_id = avg_distances.idxmin()
            min_distance = avg_distances.min()

            # Get track lengths
            ref_length = ref_corr["ref_track_length"].iloc[0]
            comp_length = ref_corr[ref_corr["comp_track_id"] == closest_comp_id][
                "comp_track_length"
            ].iloc[0]

            closest_matches.append({
                f"{reference_label}_ID": ref_id,
                f"{reference_label}_Length": ref_length,
                f"{comp_label}_ID": closest_comp_id,
                f"{comp_label}_Length": comp_length,
                "Distance": min_distance,
            })

        correspondence_tables[comp_label] = pd.DataFrame(closest_matches)

    return correspondence_tables


def quantify_particles_by_timepoint(
    df: pd.DataFrame, channel_name: str, df_name: str
) -> pd.DataFrame:
    particles_per_timepoint = df.groupby("t").size().reset_index(name="particle_count")  # type: ignore
    particles_per_timepoint["channel"] = channel_name
    print(f"Experiment name: {df_name}")
    print("=" * 50)
    print(
        f"\n Max particles in single timepoint: {particles_per_timepoint['particle_count'].max()} \n Min particles in single timepoint: {particles_per_timepoint['particle_count'].min()}"
    )
    print("=" * 50)
    return particles_per_timepoint


def find_closest_particle_per_frame(
    cme_track_data: pd.DataFrame,
    dino_df: pd.DataFrame,
    early_frames_range: int = 10,
) -> Tuple[pd.DataFrame, Optional[int]]:
    results = []
    early_frame_matches = defaultdict(list)

    for _, cme_row in cme_track_data.iterrows():
        t = cme_row["t"]
        cme_pos = cme_row[["x", "y", "z"]].values

        dino_at_t = dino_df[dino_df["t"] == t]
        if len(dino_at_t) == 0:
            print(f"No DINO particles found at time {t}")
            continue

        dino_positions = dino_at_t[["x", "y", "z"]].values
        distances = np.linalg.norm(dino_positions - cme_pos, axis=1)

        closest_idx = np.argmin(distances)
        closest_dino_row = dino_at_t.iloc[closest_idx]

        results.append({
            "t": t,
            "channel_1_x": cme_row["x"],
            "channel_1_y": cme_row["y"],
            "channel_1_z": cme_row["z"],
            "channel_1_intensity": cme_row["intensity"],
            "channel_2_id": closest_dino_row["ID"],
            "channel_2_x": closest_dino_row["x"],
            "channel_2_y": closest_dino_row["y"],
            "channel_2_z": closest_dino_row["z"],
            "channel_2_intensity": closest_dino_row["intensity"],
            "distance": distances[closest_idx],
        })

        if t <= early_frames_range:
            early_frame_matches[closest_dino_row["ID"]].append(distances[closest_idx])

    best_dino_id = (
        min(early_frame_matches.keys(), key=lambda x: np.mean(early_frame_matches[x]))  # type: ignore
        if early_frame_matches
        else None
    )

    return pd.DataFrame(results), best_dino_id


def find_closest_particle_and_analyze_track(
    main_experiment_df: pd.DataFrame,
    experiment_to_match: pd.DataFrame,
    early_frames_range: int = 10,
    channel_1_name: str = "channel_1",
    channel_2_name: str = "channel_2",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
    """Combined analysis: find closest particles AND analyze specific track."""
    frame_matches = []
    early_frame_matches = defaultdict(list)

    for _, cme_row in main_experiment_df.iterrows():
        t = cme_row["t"]
        cme_pos = cme_row[["x", "y", "z"]].values

        dino_at_t = experiment_to_match[experiment_to_match["t"] == t]
        if len(dino_at_t) == 0:
            continue

        distances = np.linalg.norm(dino_at_t[["x", "y", "z"]].values - cme_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_dino = dino_at_t.iloc[closest_idx]

        result = {
            "t": t,
            f"{channel_1_name}_x": cme_row["x"],
            f"{channel_1_name}_y": cme_row["y"],
            f"{channel_1_name}_z": cme_row["z"],
            f"{channel_1_name}_intensity": cme_row["intensity"],
            f"{channel_2_name}_id": closest_dino["ID"],
            f"{channel_2_name}_x": closest_dino["x"],
            f"{channel_2_name}_y": closest_dino["y"],
            f"{channel_2_name}_z": closest_dino["z"],
            f"{channel_2_name}_intensity": closest_dino["intensity"],
            "distance": distances[closest_idx],
            "corrected_intensity_ratio": cme_row["intensity"]
            / closest_dino["intensity"],
        }
        frame_matches.append(result)

        if t <= early_frames_range:
            early_frame_matches[closest_dino["ID"]].append(distances[closest_idx])

    frame_matches_df = pd.DataFrame(frame_matches)

    # Find best DINO ID and filter results for that specific track
    if early_frame_matches:
        best_dino_id = min(
            early_frame_matches.keys(),
            key=lambda x: np.mean(early_frame_matches[x]),  # type: ignore
        )

        # Filter frame matches to only include the best DINO track
        best_track_matches = frame_matches_df[
            frame_matches_df[f"{channel_2_name}_id"] == best_dino_id
        ].copy()

        return frame_matches_df, best_track_matches, best_dino_id

    return frame_matches_df, pd.DataFrame(), None


def calculate_weighted_rolling_average(data: np.ndarray) -> np.ndarray:
    data = np.array(data) if not isinstance(data, np.ndarray) else data
    rolling_avg = np.zeros_like(data, dtype=float)

    for i in range(len(data)):
        # Handle edge cases
        if i == 0:
            # First point: no previous point, use current and next
            if len(data) > 1:
                rolling_avg[i] = (3 * data[i] + 1 * data[i + 1]) / 4
            else:
                rolling_avg[i] = data[i]
        elif i == len(data) - 1:
            # Last point: no next point, use previous and current
            rolling_avg[i] = (1 * data[i - 1] + 3 * data[i]) / 4
        else:
            # Middle points: use previous, current, and next
            rolling_avg[i] = (1 * data[i - 1] + 3 * data[i] + 1 * data[i + 1]) / 5

    return rolling_avg


def filter_and_plot_by_distance_to_pdf_for_2_channels(
    comprehensive_data: pd.DataFrame,
    dfs_to_correct: List[pd.DataFrame],
    output_pdf_path: str = "track_analysis_results.pdf",
    n_frames_min: int = 10,
    max_distance: float = 4.0,
    plot_start: int = 0,
    plot_end: int = 10,
    ratio_scale: Tuple[float, float] = (0.018, 0.18),
) -> List[Dict[str, Any]]:
    """
    Modified version that saves all plots to a single PDF file.
    """
    available_pairs = comprehensive_data[
        ["exp_1_track_id", "exp_2_track_id_best"]
    ].drop_duplicates()

    print(available_pairs.head())
    print(
        f"Filtering {len(available_pairs)} available pairs by mean distance < {max_distance}"
    )
    print("=" * 60)

    valid_pairs = []

    # First pass: Calculate mean distance for each pair
    for i, (_, row) in enumerate(available_pairs.iterrows()):
        exp_1_track_id = row["exp_1_track_id"]
        exp_2_track_id_best = row["exp_2_track_id_best"]

        # Get data for this pair
        comparison_data = comprehensive_data[
            (comprehensive_data["exp_1_track_id"] == exp_1_track_id)
            & (comprehensive_data["exp_2_track_id_best"] == exp_2_track_id_best)
        ].copy()

        if len(comparison_data) > 0:
            mean_dist = comparison_data["distance"].mean()

            if mean_dist < max_distance and len(comparison_data) >= n_frames_min:
                valid_pairs.append({
                    "exp_1_track_id": exp_1_track_id,
                    "exp_2_track_id_best": exp_2_track_id_best,
                    "mean_distance": mean_dist,
                    "n_frames": len(comparison_data),
                })
                print(
                    f"{len(valid_pairs):3d}. exp_1_track_id={exp_1_track_id:3.0f} <-> exp_2_track_id_best={exp_2_track_id_best:4.0f} | Mean dist: {mean_dist:.2f} | Frames: {len(comparison_data)}"
                )

    # Sort valid_pairs by length of track
    valid_pairs.sort(key=lambda x: x["n_frames"], reverse=True)
    plot_to_start = min(len(valid_pairs), plot_start)
    plot_to_end = min(len(valid_pairs), plot_end)

    print(
        f"\n\n\nWe have {len(valid_pairs)} pairs, plotting pairs within index {plot_to_start} to {plot_to_end}"
    )
    print(f"Saving all plots to: {output_pdf_path}")

    # Create PDF with all plots
    with PdfPages(output_pdf_path) as pdf:
        for i, pair in enumerate(valid_pairs[plot_to_start:plot_to_end]):
            print(
                f"\nPlotting pair {i + 1}/{plot_to_end - plot_to_start}: exp_1_track_id={pair['exp_1_track_id']} <-> exp_2_track_id_best={pair['exp_2_track_id_best']} (mean dist: {pair['mean_distance']:.2f})"
            )

            # Call your plotting function but don't show the plot
            plot_max_projection_analysis_to_pdf_for_2_channels(
                exp_1_track_id=pair["exp_1_track_id"],
                exp_2_track_id_best=pair["exp_2_track_id_best"],
                comprehensive_data=comprehensive_data,
                dfs_to_correct=dfs_to_correct,
                ratio_scale=ratio_scale,
                pdf=pdf,
            )
    print(f"\nAll plots saved to: {output_pdf_path}")
    return valid_pairs


def plot_max_projection_analysis_to_pdf_for_2_channels(
    exp_1_track_id: int,
    exp_2_track_id_best: int,
    comprehensive_data: pd.DataFrame,
    dfs_to_correct: List[pd.DataFrame],
    ratio_scale: Tuple[float, float] = (0.0, 0.3),
    pdf: Optional[PdfPages] = None,
) -> None:
    comparison_data = comprehensive_data[
        (comprehensive_data["exp_1_track_id"] == exp_1_track_id)
        & (comprehensive_data["exp_2_track_id_best"] == exp_2_track_id_best)
    ].copy()
    comparison_data = comparison_data.sort_values("t")
    comparison_data = correct_intensities_for_2_channels(
        comparison_data=comparison_data,
        dfs_to_correct=dfs_to_correct,
        exp_1_track_id=exp_1_track_id,
        exp_2_track_id_best=exp_2_track_id_best,
    )
    # Get decay correction for DINO
    # Update intensity ratio calculation
    comparison_data["corrected_intensity_ratio"] = np.minimum(
        comparison_data["channel_1_intensity_corrected"],
        comparison_data["channel_2_intensity_corrected"],
    ) / np.maximum(
        comparison_data["channel_1_intensity_corrected"],
        comparison_data["channel_2_intensity_corrected"],
    )

    # Calculate rolling averages for intensity data
    cme_rolling_avg = calculate_weighted_rolling_average(
        np.array(comparison_data["channel_1_intensity_corrected"].values)
    )
    dino_rolling_avg = calculate_weighted_rolling_average(
        np.array(comparison_data["channel_2_intensity_corrected"].values)
    )
    dino_raw_rolling_avg = calculate_weighted_rolling_average(
        np.array(comparison_data["channel_2_intensity_corrected"].values)
    )

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Plot 1: Combined Intensity Comparison (with rolling averages)
    ax1 = axes[0, 0]
    ax1.plot(
        comparison_data["t"],
        comparison_data["channel_1_intensity_corrected"],
        "r-",
        label=f"exp_1_track_id={exp_1_track_id} (normalized)",
        marker="o",
        alpha=0.6,
    )
    ax1.plot(
        comparison_data["t"],
        comparison_data["channel_2_intensity_corrected"],
        "b-",
        label=f"exp_2_track_id_best={exp_2_track_id_best} (corrected)",
        marker="s",
        alpha=0.6,
    )
    ax1.plot(
        comparison_data["t"],
        comparison_data["channel_2_intensity_corrected"],
        "c-",
        label=f"exp_2_track_id_best={exp_2_track_id_best} (raw)",
        marker="^",
        linewidth=2,
        alpha=0.6,
    )

    # Add rolling averages as thicker overlays
    ax1.plot(
        comparison_data["t"],
        cme_rolling_avg,
        "r-",
        label=f"exp_1_track_id={exp_1_track_id} (rolling avg)",
        linewidth=3,
        alpha=0.9,
    )
    ax1.plot(
        comparison_data["t"],
        dino_rolling_avg,
        "b-",
        label=f"exp_2_track_id_best={exp_2_track_id_best} (corrected rolling avg)",
        linewidth=3,
        alpha=0.9,
    )
    ax1.plot(
        comparison_data["t"],
        dino_raw_rolling_avg,
        "c-",
        label=f"exp_2_track_id_best={exp_2_track_id_best} (raw rolling avg)",
        linewidth=3,
        alpha=0.9,
    )

    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Normalized Intensity")
    ax1.set_title("Combined Intensity Comparison (with Rolling Averages)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Distance over time
    ax2 = axes[0, 1]
    ax2.plot(
        comparison_data["t"],
        comparison_data["distance"],
        "g-",
        marker="d",
    )
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Distance (units)")
    ax2.set_title("Distance Over Time")
    ax2.grid(True)

    # Add statistics lines
    mean_dist = comparison_data["distance"].mean()
    std_dist = comparison_data["distance"].std()
    ax2.axhline(
        y=mean_dist,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_dist:.2f}",
    )
    ax2.axhline(
        y=mean_dist + std_dist,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Mean + σ: {mean_dist + std_dist:.2f}",
    )
    ax2.axhline(
        y=mean_dist - std_dist,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Mean - σ: {mean_dist - std_dist:.2f}",
    )
    ax2.legend()

    # Plot 3: Corrected intensity ratio over time
    ax3 = axes[0, 2]
    ax3.plot(
        comparison_data["t"],
        comparison_data["corrected_intensity_ratio"],
        "m-",
        linewidth=2,
        marker="v",
        markersize=4,
    )
    ax3.axhline(y=1, color="white", linestyle="--", alpha=0.7, label="Ratio = 1")
    mean_ratio = comparison_data["corrected_intensity_ratio"].mean()
    ax3.axhline(
        y=mean_ratio,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_ratio:.3f}",
    )
    ax3.set_xlabel("Time (frames)")
    ax3.set_ylabel("Intensity Ratio (Min/Max)")
    ax3.set_title("Corrected Intensity Ratio Over Time")
    ax3.set_ylim(ratio_scale[0], ratio_scale[1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: 3D trajectory comparison
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax4.plot(
        comparison_data["channel_1_x"],
        comparison_data["channel_1_y"],
        comparison_data["channel_1_z"],
        "r-",
        label=f"exp_1_track_id={exp_1_track_id}",
        marker="o",
        alpha=0.7,
    )
    ax4.plot(
        comparison_data["channel_2_x"],
        comparison_data["channel_2_y"],
        comparison_data["channel_2_z"],
        "b-",
        linewidth=2,
        label=f"exp_2_track_id_best={exp_2_track_id_best}",
        marker="s",
        markersize=3,
        alpha=0.7,
    )
    ax4.set_xlabel("X Position")
    ax4.set_ylabel("Y Position")
    ax4.set_title("3D Trajectory Comparison (X-Y-Z)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: CME intensity only (with rolling average)
    ax5 = axes[1, 1]
    ax5.plot(
        comparison_data["t"],
        comparison_data["channel_1_intensity_corrected"],
        "r-",
        label=f"exp_1_track_id={exp_1_track_id} (normalized)",
        marker="o",
        linewidth=2,
        alpha=0.6,
    )
    ax5.plot(
        comparison_data["t"],
        cme_rolling_avg,
        "r-",
        label=f"exp_1_track_id={exp_1_track_id} (rolling avg)",
        linewidth=4,
        alpha=0.9,
    )
    ax5.set_xlabel("Time (frames)")
    ax5.set_ylabel("Normalized CME Intensity")
    ax5.set_title("CME Intensity Only (with Rolling Average)")
    ax5.legend()
    ax5.grid(True)

    # Plot 6: DINO intensity only (with rolling average)
    ax6 = axes[1, 2]
    ax6.plot(
        comparison_data["t"],
        comparison_data["channel_2_intensity_corrected"],
        "b-",
        label=f"exp_2_track_id_best={exp_2_track_id_best} (corrected)",
        marker="s",
        linewidth=2,
        alpha=0.6,
    )
    ax6.plot(
        comparison_data["t"],
        dino_rolling_avg,
        "b-",
        label=f"exp_2_track_id_best={exp_2_track_id_best} (rolling avg)",
        linewidth=4,
        alpha=0.9,
    )
    ax6.set_xlabel("Time (frames)")
    ax6.set_ylabel("Normalized DINO Intensity")
    ax6.set_title("DINO Intensity Only (with Rolling Average)")
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()

    # Save to PDF instead of showing
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")

    plt.show()
    # Close the figure to free memory
    plt.close(fig)

    # Print detailed summary (same as before)
    print("\n" + "=" * 60)
    print("TRACK ANALYSIS RESULTS (WITH ROLLING AVERAGES)")
    print("=" * 60)
    print(f"exp_1_track_id={exp_1_track_id}")
    print(f"exp_2_track_id_best={exp_2_track_id_best}")
    print(f"Common timeframes analyzed: {len(comparison_data)}")
    print(f"Time range: {comparison_data['t'].min()} to {comparison_data['t'].max()}")
    print("\nRolling Average Details:")
    print(
        "  Weight scheme: Current point (3) + Previous point (1) + Next point (1) = Total weight (5)"
    )
    print("  Edge handling: First/last points use adjusted weights")
    print("\nSpatial Analysis:")
    print(
        f"  exp_1_track_id={exp_1_track_id} position range: X({comparison_data['channel_1_x'].min():.1f}-{comparison_data['channel_1_x'].max():.1f}), "
        + f"Y({comparison_data['channel_1_y'].min():.1f}-{comparison_data['channel_1_y'].max():.1f}), "
        + f"Z({comparison_data['channel_1_z'].min():.1f}-{comparison_data['channel_1_z'].max():.1f})"
    )
    print(
        f"  exp_2_track_id_best={exp_2_track_id_best} position range: X({comparison_data['channel_2_x'].min():.1f}-{comparison_data['channel_2_x'].max():.1f}), "
        + f"Y({comparison_data['channel_2_y'].min():.1f}-{comparison_data['channel_2_y'].max():.1f}), "
        + f"Z({comparison_data['channel_2_z'].min():.1f}-{comparison_data['channel_2_z'].max():.1f})"
    )
    print("=" * 60)


def find_proximate_dino_tracks(
    exp_1_track_id: int,
    experiment_1_df: pd.DataFrame,
    experiment_2_df: pd.DataFrame,
    proximity_threshold: float = 3.0,
) -> Tuple[Dict[int, List[int]], set]:
    exp_1_track = experiment_1_df[experiment_1_df["ID"] == exp_1_track_id]
    if len(exp_1_track) == 0:
        raise ValueError(f"Track {exp_1_track_id} not found in experiment 1")

    # find time range
    t_min, t_max = int(exp_1_track["t"].min()), int(exp_1_track["t"].max())
    print(f"Time range: {t_min} to {t_max}")
    # all close tracks in a set
    proximate_by_time = defaultdict(list)
    proximate_exp2_tracks = set()
    for _, exp_1_point in exp_1_track.iterrows():
        t = exp_1_point["t"]
        exp_1_pos = np.array([exp_1_point["x"], exp_1_point["y"], exp_1_point["z"]])

        # we look at each time point from exp1 and find all tracks in exp2 that are close to it
        exp_2_at_t = experiment_2_df[experiment_2_df["t"] == t]
        for _, exp_2_point in exp_2_at_t.iterrows():
            exp_2_pos = np.array([exp_2_point["x"], exp_2_point["y"], exp_2_point["z"]])
            distance = np.linalg.norm(exp_1_pos - exp_2_pos)
            if distance <= proximity_threshold:
                proximate_by_time[t].append(exp_2_point["ID"])
                proximate_exp2_tracks.add(exp_2_point["ID"])

    return proximate_by_time, proximate_exp2_tracks


def closest_detections_per_frame(
    experiment_1_df: pd.DataFrame,
    experiment_2_df: pd.DataFrame,
    dist_func: Callable,
    max_distance: float = 3.0,
) -> Dict[int, Dict[str, Any]]:
    exp1_coords, exp2_coords = (
        experiment_1_df[["x", "y", "x"]].values,
        experiment_2_df[["x", "y", "x"]].values,
    )
    distances = dist_func(exp1_coords, exp2_coords)
    print(
        f"The shape of the distances is {distances.shape} \n The shape of the exp1_coords is {exp1_coords.shape} \n The shape of the exp2_coords is {exp2_coords.shape}"
    )
    closest_exp2_detection = {}
    for i, (_, curr_row) in tqdm(
        enumerate(experiment_1_df.iterrows()), total=len(experiment_1_df)
    ):
        # curr_id = curr_row["t"]
        curr_distances = distances[i]
        sorted_indices = np.argsort(curr_distances)
        valid_indices = sorted_indices[curr_distances[sorted_indices] <= max_distance]
        closest_exp2_detection[i] = {}
        closest_exp2_detection[i]["valid_indices"] = valid_indices
        for idx in valid_indices:
            exp2_row = experiment_2_df.iloc[idx]
            closest_exp2_detection[i]["exp2_coords"] = exp2_row[["x", "y", "z"]].values
            closest_exp2_detection[i]["exp1_coords"] = curr_row[["x", "y", "z"]].values
            # add time as well
            # closest_exp2_detection[i]["exp1_time"] = curr_row["t"]
            # closest_exp2_detection[i]["exp2_time"] = exp2_row["t"]
            closest_exp2_detection[i]["distance"] = curr_distances[idx]
    return closest_exp2_detection


def find_unmatched_particles(
    experiment_1_df: pd.DataFrame,
    experiment_2_df: pd.DataFrame,
    closest_exp2_detection: Dict[int, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    unmatched_exp1_particles = []
    matched_exp2_indices = set()
    for i, detection_info in tqdm(
        closest_exp2_detection.items(), total=len(closest_exp2_detection)
    ):
        valid_indices = detection_info["valid_indices"]
        if len(valid_indices) == 0:
            exp1_row = experiment_1_df.iloc[i]
            unmatched_exp1_particles.append({
                "exp1_index": i,
                "exp1_id": exp1_row["ID"],
                "coordinates": (exp1_row["x"], exp1_row["y"], exp1_row["z"]),
            })
        else:
            matched_exp2_indices.update(valid_indices)
    total_exp_1_particles = len(experiment_1_df)
    total_exp_2_particles = len(experiment_2_df)
    matched_exp_particles = total_exp_1_particles - len(unmatched_exp1_particles)
    all_exp2_indices = set(range(total_exp_2_particles))
    unmatched_exp2_indices = all_exp2_indices - matched_exp2_indices
    print(
        f"Number of unmatched exp1 particles: {total_exp_1_particles}, unmatched exp1: {len(unmatched_exp1_particles)}, matched exp1: {matched_exp_particles}"
    )
    print(
        f"Number of unmatched exp2 particles: {total_exp_2_particles}, unmatched exp2: {total_exp_2_particles - len(matched_exp2_indices)}, matched exp2: {len(matched_exp2_indices)}"
    )
    return (
        unmatched_exp1_particles,
        sorted(matched_exp2_indices),
        sorted(unmatched_exp2_indices),
    )


#################################### 3 channel functions ####################################


def filter_and_plot_by_distance_to_pdf_for_3_channels(
    comprehensive_data: pd.DataFrame,
    dfs_to_correct: List[pd.DataFrame],
    output_pdf_path: str = "track_analysis_results.pdf",
    n_frames_min: int = 10,
    max_distance: float = 4.0,
    plot_start: int = 0,
    plot_end: int = 10,
    ratio_scale: Tuple[float, float] = (0.018, 0.18),
    channel_1_name: str = "560",
    channel_2_name: str = "640",
    channel_3_name: str = "488",
) -> List[Dict[str, Any]]:
    """
    Modified version that saves all plots to a single PDF file.
    """
    available_pairs = comprehensive_data[
        ["exp_1_track_id", "exp_2_track_id_best", "exp_3_track_id_best"]
    ].drop_duplicates()

    print(available_pairs.head())
    print(
        f"Filtering {len(available_pairs)} available pairs by mean distance < {max_distance}"
    )
    print("=" * 60)

    valid_pairs = []

    # First pass: Calculate mean distance for each pair
    for i, (_, row) in enumerate(available_pairs.iterrows()):
        exp_1_track_id = row["exp_1_track_id"]
        exp_2_track_id_best = row["exp_2_track_id_best"]
        exp_3_track_id_best = row["exp_3_track_id_best"]
        # Get data for this pair
        comparison_data = comprehensive_data[
            (comprehensive_data["exp_1_track_id"] == exp_1_track_id)
            & (comprehensive_data["exp_2_track_id_best"] == exp_2_track_id_best)
            & (comprehensive_data["exp_3_track_id_best"] == exp_3_track_id_best)
        ].copy()

        if len(comparison_data) > 0:
            mean_dist = comparison_data["distance_channel_2"].mean()
            mean_dist_chan3 = comparison_data["distance_channel_3"].mean()
            mean_dist_chan23 = comparison_data["distance_channel_23"].mean()
            if mean_dist < max_distance and len(comparison_data) >= n_frames_min:
                valid_pairs.append({
                    "exp_1_track_id": exp_1_track_id,
                    "exp_2_track_id_best": exp_2_track_id_best,
                    "exp_3_track_id_best": exp_3_track_id_best,
                    "mean_distance_channel_2": mean_dist,
                    "mean_distance_channel_3": mean_dist_chan3,
                    "mean_distance_channel_23": mean_dist_chan23,
                    "n_frames": len(comparison_data),
                })
                print(
                    f"{len(valid_pairs):3d}. exp_1_track_id={exp_1_track_id:3.0f} <-> exp_2_track_id_best={exp_2_track_id_best:4.0f} <-> exp_3_track_id_best={exp_3_track_id_best:4.0f} | Mean dist: {mean_dist:.2f} | Mean dist chan3: {mean_dist_chan3:.2f} | Frames: {len(comparison_data)}"
                )

    # Sort valid_pairs by length of track
    valid_pairs.sort(key=lambda x: x["n_frames"], reverse=True)
    plot_to_start = min(len(valid_pairs), plot_start)
    plot_to_end = min(len(valid_pairs), plot_end)

    print(
        f"\n\n\nWe have {len(valid_pairs)} pairs, plotting pairs within index {plot_to_start} to {plot_to_end}"
    )
    print(f"Saving all plots to: {output_pdf_path}")

    # Create PDF with all plots
    with PdfPages(output_pdf_path) as pdf:
        for i, pair in enumerate(valid_pairs[plot_to_start:plot_to_end]):
            print(
                f"\nPlotting pair {i + 1}/{plot_to_end - plot_to_start}: exp_1_track_id={pair['exp_1_track_id']} <-> exp_2_track_id_best={pair['exp_2_track_id_best']} <-> exp_3_track_id_best={pair['exp_3_track_id_best']} (mean dist: {pair['mean_distance_channel_2']:.2f}, mean dist chan3: {pair['mean_distance_channel_3']:.2f})"
            )

            # Call your plotting function but don't show the plot
            plot_max_projection_analysis_to_pdf_for_3_channels(
                exp_1_track_id=pair["exp_1_track_id"],
                exp_2_track_id_best=pair["exp_2_track_id_best"],
                exp_3_track_id_best=pair["exp_3_track_id_best"],
                comprehensive_data=comprehensive_data,
                dfs_to_correct=dfs_to_correct,
                ratio_scale=ratio_scale,
                channel_1_name=channel_1_name,
                channel_2_name=channel_2_name,
                channel_3_name=channel_3_name,
                pdf=pdf,
            )
    print(f"\nAll plots saved to: {output_pdf_path}")
    return valid_pairs


def correct_intensities_for_2_channels(
    comparison_data: pd.DataFrame,
    dfs_to_correct: List[pd.DataFrame],
    exp_1_track_id: int,
    exp_2_track_id_best: int,
) -> pd.DataFrame:
    for i in range(len(dfs_to_correct)):
        # Use appropriate track ID for each channel
        if i == 0:
            track_id = exp_1_track_id
        elif i == 1:
            track_id = exp_2_track_id_best

        decay_params = get_decay_correction_for_dino_track(
            dino_df=dfs_to_correct[i],
            dino_id=track_id,  # type: ignore
        )
        k_value = decay_params["k"]
        fit_success = decay_params["fit_success"]

        if fit_success and not np.isnan(k_value) and k_value > 0:
            corrected_intensities = apply_decay_correction_to_intensities(
                time_points=comparison_data["t"].values,
                intensities=comparison_data[f"channel_{i + 1}_intensity"].values.astype(
                    float
                ),
                k_value=k_value,
            )
            comparison_data[f"channel_{i + 1}_intensity_corrected"] = (
                corrected_intensities
            )
        else:
            comparison_data[f"channel_{i + 1}_intensity_corrected"] = comparison_data[
                f"channel_{i + 1}_intensity"
            ]
    return comparison_data


def correct_intensities_for_3_channels(
    comparison_data: pd.DataFrame,
    dfs_to_correct: List[pd.DataFrame],
    exp_1_track_id: int,
    exp_2_track_id_best: int,
    exp_3_track_id_best: int,
) -> pd.DataFrame:
    for i in range(len(dfs_to_correct)):
        # Use appropriate track ID for each channel
        if i == 0:
            track_id = exp_1_track_id
        elif i == 1:
            track_id = exp_2_track_id_best
        else:
            track_id = exp_3_track_id_best

        decay_params = get_decay_correction_for_dino_track(
            dino_df=dfs_to_correct[i], dino_id=track_id
        )
        k_value = decay_params["k"]
        fit_success = decay_params["fit_success"]

        if fit_success and not np.isnan(k_value) and k_value > 0:
            corrected_intensities = apply_decay_correction_to_intensities(
                time_points=comparison_data["t"].values,
                intensities=comparison_data[f"channel_{i + 1}_intensity"].values.astype(
                    float
                ),
                k_value=k_value,
            )
            comparison_data[f"channel_{i + 1}_intensity_corrected"] = (
                corrected_intensities
            )
        else:
            comparison_data[f"channel_{i + 1}_intensity_corrected"] = comparison_data[
                f"channel_{i + 1}_intensity"
            ]
    return comparison_data


def plot_max_projection_analysis_to_pdf_for_3_channels(
    exp_1_track_id: int,
    exp_2_track_id_best: int,
    exp_3_track_id_best: int,
    comprehensive_data: pd.DataFrame,
    dfs_to_correct: List[pd.DataFrame],
    ratio_scale: Tuple[float, float] = (0.0, 0.3),
    channel_1_name: str = "560",
    channel_2_name: str = "640",
    channel_3_name: str = "488",
    pdf: Optional[PdfPages] = None,
) -> None:
    # Filter data for the specific track combination
    comparison_data = comprehensive_data[
        (comprehensive_data["exp_1_track_id"] == exp_1_track_id)
        & (comprehensive_data["exp_2_track_id_best"] == exp_2_track_id_best)
        & (comprehensive_data["exp_3_track_id_best"] == exp_3_track_id_best)
    ].copy()
    comparison_data = comparison_data.sort_values("t")
    comparison_data = correct_intensities_for_3_channels(
        comparison_data=comparison_data,
        dfs_to_correct=dfs_to_correct,
        exp_1_track_id=exp_1_track_id,
        exp_2_track_id_best=exp_2_track_id_best,
        exp_3_track_id_best=exp_3_track_id_best,
    )

    # Calculate rolling averages for all channels
    channel_1_rolling_avg = calculate_weighted_rolling_average(
        np.array(comparison_data["channel_1_intensity_corrected"].values)
    )
    channel_2_rolling_avg = calculate_weighted_rolling_average(
        np.array(comparison_data["channel_2_intensity_corrected"].values)
    )
    channel_3_rolling_avg = calculate_weighted_rolling_average(
        np.array(comparison_data["channel_3_intensity_corrected"].values)
    )

    # Calculate intensity ratios
    comparison_data["ratio_ch1_ch2"] = np.minimum(
        comparison_data["channel_1_intensity_corrected"],
        comparison_data["channel_2_intensity_corrected"],
    ) / np.maximum(
        comparison_data["channel_1_intensity_corrected"],
        comparison_data["channel_2_intensity_corrected"],
    )

    comparison_data["ratio_ch1_ch3"] = np.minimum(
        comparison_data["channel_1_intensity_corrected"],
        comparison_data["channel_3_intensity_corrected"],
    ) / np.maximum(
        comparison_data["channel_1_intensity_corrected"],
        comparison_data["channel_3_intensity_corrected"],
    )

    comparison_data["ratio_ch2_ch3"] = np.minimum(
        comparison_data["channel_2_intensity_corrected"],
        comparison_data["channel_3_intensity_corrected"],
    ) / np.maximum(
        comparison_data["channel_2_intensity_corrected"],
        comparison_data["channel_3_intensity_corrected"],
    )

    # Scale Channel 3 intensity to match the Ch1/Ch2 ratio scale for comparison
    ch3_scaled = (
        comparison_data["channel_3_intensity_corrected"]
        / comparison_data["channel_3_intensity_corrected"].max()
        * ratio_scale[1]
    )

    # Create the figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))

    # Plot 1: Combined Intensity Comparison - All 3 channels overlaid
    ax1 = axes[0, 0]
    ax1.plot(
        comparison_data["t"],
        comparison_data["channel_1_intensity_corrected"],
        "r-",
        label=f"{channel_1_name} (Track {exp_1_track_id}) - Corrected",
        marker="o",
        alpha=0.6,
    )
    ax1.plot(
        comparison_data["t"],
        comparison_data["channel_2_intensity_corrected"],
        "b-",
        label=f"{channel_2_name} (Track {exp_2_track_id_best}) - Corrected",
        marker="s",
        alpha=0.6,
    )
    ax1.plot(
        comparison_data["t"],
        comparison_data["channel_3_intensity_corrected"],
        "g-",
        label=f"{channel_3_name} (Track {exp_3_track_id_best}) - Corrected",
        marker="^",
        alpha=0.6,
    )

    # Add rolling averages as thicker overlays
    ax1.plot(
        comparison_data["t"],
        channel_1_rolling_avg,
        "r-",
        label=f"{channel_1_name} (Rolling Avg)",
        linewidth=3,
        alpha=0.9,
    )
    ax1.plot(
        comparison_data["t"],
        channel_2_rolling_avg,
        "b-",
        label=f"{channel_2_name} (Rolling Avg)",
        linewidth=3,
        alpha=0.9,
    )
    ax1.plot(
        comparison_data["t"],
        channel_3_rolling_avg,
        "g-",
        label=f"{channel_3_name} (Rolling Avg)",
        linewidth=3,
        alpha=0.9,
    )

    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Normalized Intensity")
    ax1.set_title("3-Channel Intensity Overlay (Corrected + Rolling Averages)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Distance over time (Channel 2)
    ax2 = axes[0, 1]
    ax2.plot(
        comparison_data["t"],
        comparison_data["distance_channel_2"],
        "b-",
        marker="d",
        label=f"{channel_1_name} vs {channel_2_name}",
    )
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Distance (units)")
    ax2.set_title(f"Distance Over Time - {channel_1_name} vs {channel_2_name}")
    ax2.grid(True)

    mean_dist_2 = comparison_data["distance_channel_2"].mean()
    std_dist_2 = comparison_data["distance_channel_2"].std()
    ax2.axhline(
        y=mean_dist_2,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_dist_2:.2f}",
    )
    ax2.legend()

    # Plot 3: Distance over time (Channel 3)
    ax3 = axes[0, 2]
    ax3.plot(
        comparison_data["t"],
        comparison_data["distance_channel_23"],
        "g-",
        marker="d",
        label=f"{channel_2_name} vs {channel_3_name}",
    )
    ax3.set_xlabel("Time (frames)")
    ax3.set_ylabel("Distance (units)")
    ax3.set_title(f"Distance Over Time - {channel_2_name} vs {channel_3_name}")
    ax3.grid(True)

    mean_dist_23 = comparison_data["distance_channel_23"].mean()
    std_dist_23 = comparison_data["distance_channel_23"].std()
    ax3.axhline(
        y=mean_dist_23,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_dist_23:.2f}",
    )
    ax3.legend()

    # Plot 4: Individual Channel 1 Intensity (moved from position 5)
    ax4 = axes[1, 0]
    ax4.plot(
        comparison_data["t"],
        comparison_data["channel_1_intensity_corrected"],
        "r-",
        label=f"{channel_1_name} (Track {exp_1_track_id}) - Corrected",
        marker="o",
        linewidth=2,
        alpha=0.6,
    )
    ax4.plot(
        comparison_data["t"],
        channel_1_rolling_avg,
        "r-",
        label="Rolling Average",
        linewidth=4,
        alpha=0.9,
    )
    ax4.set_xlabel("Time (frames)")
    ax4.set_ylabel("Normalized Intensity")
    ax4.set_title(f"{channel_1_name} Intensity (Corrected)")
    ax4.legend()
    ax4.grid(True)

    # Plot 5: Individual Channel 2 Intensity (moved from position 6)
    ax5 = axes[1, 1]
    ax5.plot(
        comparison_data["t"],
        comparison_data["channel_2_intensity_corrected"],
        "b-",
        label=f"{channel_2_name} (Track {exp_2_track_id_best}) - Corrected",
        marker="s",
        linewidth=2,
        alpha=0.6,
    )
    ax5.plot(
        comparison_data["t"],
        channel_2_rolling_avg,
        "b-",
        label="Rolling Average",
        linewidth=4,
        alpha=0.9,
    )
    ax5.set_xlabel("Time (frames)")
    ax5.set_ylabel("Normalized Intensity")
    ax5.set_title(f"{channel_2_name} Intensity (Corrected)")
    ax5.legend()
    ax5.grid(True)

    # Plot 6: Individual Channel 3 Intensity (moved from position 7)
    ax6 = axes[1, 2]
    ax6.plot(
        comparison_data["t"],
        comparison_data["channel_3_intensity_corrected"],
        "g-",
        label=f"{channel_3_name} (Track {exp_3_track_id_best}) - Corrected",
        marker="^",
        linewidth=2,
        alpha=0.6,
    )
    ax6.plot(
        comparison_data["t"],
        channel_3_rolling_avg,
        "g-",
        label="Rolling Average",
        linewidth=4,
        alpha=0.9,
    )
    ax6.set_xlabel("Time (frames)")
    ax6.set_ylabel("Normalized Intensity")
    ax6.set_title(f"{channel_3_name} Intensity (Corrected)")
    ax6.legend()
    ax6.grid(True)

    # Plot 7: Intensity Ratios - Channel 1 vs 2 (moved from position 8)
    ax7 = axes[2, 0]
    ax7.plot(
        comparison_data["t"],
        comparison_data["ratio_ch1_ch2"],
        "purple",
        linewidth=2,
        marker="v",
        markersize=4,
        label=f"{channel_1_name}/{channel_2_name} Ratio",
    )
    ax7.axhline(y=1, color="white", linestyle="--", alpha=0.7, label="Ratio = 1")
    mean_ratio_12 = comparison_data["ratio_ch1_ch2"].mean()
    ax7.axhline(
        y=mean_ratio_12,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Mean {channel_1_name}/{channel_2_name}: {mean_ratio_12:.3f}",
    )

    ax7.set_xlabel("Time (frames)")
    ax7.set_ylabel("Intensity Ratio (Min/Max)")
    ax7.set_title(f"{channel_1_name}/{channel_2_name} Ratio")
    ax7.set_ylim(ratio_scale[0], ratio_scale[1])
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: 3D trajectory comparison - All channels (MOVED TO BOTTOM ROW, SECOND COLUMN)
    ax8 = fig.add_subplot(3, 3, 8, projection="3d")
    ax8.plot(
        comparison_data["channel_1_x"],
        comparison_data["channel_1_y"],
        comparison_data["channel_1_z"],
        "r-",
        label=f"{channel_1_name} (Track {exp_1_track_id})",
        marker="o",
        alpha=0.7,
    )
    ax8.plot(
        comparison_data["channel_2_x"],
        comparison_data["channel_2_y"],
        comparison_data["channel_2_z"],
        "b-",
        label=f"{channel_2_name} (Track {exp_2_track_id_best})",
        marker="s",
        alpha=0.7,
    )
    ax8.plot(
        comparison_data["channel_3_x"],
        comparison_data["channel_3_y"],
        comparison_data["channel_3_z"],
        "g-",
        label=f"{channel_3_name} (Track {exp_3_track_id_best})",
        marker="^",
        alpha=0.7,
    )
    ax8.set_xlabel("X Position")
    ax8.set_ylabel("Y Position")
    ax8.set_zlabel("Z Position")  # type: ignore
    ax8.set_title("3D Trajectory Comparison (All Channels)")
    ax8.legend()

    # Plot 9: NEW - Ch1/Ch2 Ratio vs Scaled Channel 3 Intensity (BOTTOM ROW, THIRD COLUMN)
    ax9 = axes[2, 2]
    ax9.plot(
        comparison_data["t"],
        comparison_data["ratio_ch1_ch2"],
        "purple",
        linewidth=2,
        marker="v",
        markersize=4,
        label=f"{channel_1_name}/{channel_2_name} Ratio",
        alpha=0.8,
    )
    ax9.plot(
        comparison_data["t"],
        ch3_scaled,
        "orange",
        linewidth=2,
        marker="^",
        markersize=4,
        label=f"{channel_3_name} (Scaled to Ratio Range)",
        alpha=0.8,
    )

    # Add mean lines
    ax9.axhline(
        y=mean_ratio_12,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Mean {channel_1_name}/{channel_2_name}: {mean_ratio_12:.3f}",
    )
    mean_ch3_scaled = ch3_scaled.mean()
    ax9.axhline(
        y=mean_ch3_scaled,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean {channel_3_name} (scaled): {mean_ch3_scaled:.3f}",
    )

    ax9.set_xlabel("Time (frames)")
    ax9.set_ylabel("Scaled Intensity / Ratio")
    ax9.set_title(
        f"{channel_1_name}/{channel_2_name} Ratio vs Scaled {channel_3_name} Intensity"
    )
    ax9.set_ylim(ratio_scale[0], ratio_scale[1])
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to PDF if provided
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")

    plt.show()
    # Close the figure to free memory
    plt.close(fig)

    # Print detailed summary
    print("\n" + "=" * 80)
    print("3-CHANNEL TRACK ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Channel 1 Track ID: {exp_1_track_id} (Corrected)")
    print(f"Channel 2 Track ID: {exp_2_track_id_best} (Corrected)")
    print(
        f"Channel 3 Track ID: {exp_3_track_id_best} (Main Experiment - {channel_1_name})"
    )
    print(f"Common timeframes analyzed: {len(comparison_data)}")
    print(f"Time range: {comparison_data['t'].min()} to {comparison_data['t'].max()}")
    print("\nChannel Configuration:")
    print("  - Channels 1 & 2: Comparison channels with decay correction applied")
    print(
        f"  - Channel 3: Main experiment ({channel_1_name} channel) - no correction needed"
    )
    print("\nDistance Analysis:")
    print(f"  Channel 1 vs 2 - Mean distance: {mean_dist_2:.2f} ± {std_dist_2:.2f}")
    print(f"  Channel 2 vs 3 - Mean distance: {mean_dist_23:.2f} ± {std_dist_23:.2f}")
    print("\nIntensity Ratio Analysis:")
    print(f"  Channel 1/2 ratio: {mean_ratio_12:.3f}")
    print(f"  Channel 3 (scaled to ratio range): {mean_ch3_scaled:.3f}")
    print("=" * 80)


def find_best_particle_match(
    closest_particles: Dict[str, Any],
    feature_columns: List[str] = [],
    min_similarity_threshold: float = 0.7,
    spatial_closest_id: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simple function to find the best particle match using cosine similarity.

    Args:
        closest_particles: Dictionary containing t0 and t1 particle data
        feature_columns: List of feature column names to use
        min_similarity_threshold: Minimum cosine similarity to consider a match
        spatial_closest_id: ID of spatially closest particle for comparison
        verbose: Whether to print detailed results

    Returns:
        Dictionary with the best match and ranking information
    """
    ranking_results = rank_particles_by_cosine_similarity(
        closest_particles=closest_particles,
        feature_columns=feature_columns,
        min_similarity_threshold=min_similarity_threshold,
    )

    if "error" in ranking_results:
        return ranking_results

    best_match = ranking_results["best_match"]

    if verbose:
        print(f"\n🎯 BEST PARTICLE MATCH:")
        print(
            f"  Feature-based winner: Particle {best_match['particle_id']} "
            f"(similarity: {best_match['cosine_similarity']:.6f})"
        )

        if spatial_closest_id:
            spatial_match = next(
                (
                    p
                    for p in ranking_results["rankings"]
                    if p["particle_id"] == spatial_closest_id
                ),
                None,
            )
            if spatial_match:
                print(
                    f"  Spatial closest: Particle {spatial_closest_id} "
                    f"(similarity: {spatial_match['cosine_similarity']:.6f}, "
                    f"rank: {spatial_match['rank']})"
                )

                if best_match["particle_id"] == spatial_closest_id:
                    print("  ✅ Feature and spatial matches agree!")
                else:
                    gap = (
                        best_match["cosine_similarity"]
                        - spatial_match["cosine_similarity"]
                    )
                    print(f"  ⚠️  Feature and spatial matches differ (gap: {gap:.6f})")

        print(
            f"  Clear winner: {'Yes' if ranking_results['is_clear_winner'] else 'No'}"
        )
        print(f"  Meets threshold: {'Yes' if best_match['meets_threshold'] else 'No'}")

    return {
        "best_particle_id": best_match["particle_id"],
        "best_similarity": best_match["cosine_similarity"],
        "is_reliable": ranking_results["is_clear_winner"]
        and best_match["meets_threshold"],
        "full_ranking": ranking_results["rankings"],
        "spatial_agrees": spatial_closest_id == best_match["particle_id"]
        if spatial_closest_id
        else None,
    }


def rank_particles_by_cosine_similarity(
    closest_particles: Dict[str, Any],
    feature_columns: List[str] = [],
    min_similarity_threshold: float = 0.7,
    use_weighted_features: bool = False,
    feature_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Rank particles by cosine similarity for better particle separation.

    Args:
        closest_particles: Dictionary containing t0 and t1 particle data
        feature_columns: List of feature column names to use
        min_similarity_threshold: Minimum cosine similarity to consider a match
        use_weighted_features: Whether to apply feature weights
        feature_weights: Optional weights for features (must match feature count)

    Returns:
        Dictionary with ranking results and confidence scores
    """
    closest_particles_lst = closest_particles["t1"]
    particles_df = closest_particles_lst["particles"]

    if not feature_columns:
        feature_columns = [
            col for col in particles_df.columns if col.startswith("feature_")
        ]

    # Get reference features from t0
    if "particles" in closest_particles["t0"]:
        particles_t0 = closest_particles["t0"]["particles"]
        if len(particles_t0) == 0:
            features_t0 = closest_particles["t0"]["features"]
            if len(features_t0) == 0:
                return {"error": "No reference features found"}
            reference_features = features_t0[0]
        else:
            reference_features = particles_t0[feature_columns].iloc[0].values
    else:
        features_t0 = closest_particles["t0"]["features"]
        if len(features_t0) == 0:
            return {"error": "No reference features found"}
        reference_features = features_t0[0]

    # Get candidate features from t1
    candidate_features = particles_df[feature_columns].values
    particle_ids = particles_df.index.values

    # Apply feature weights if specified
    if use_weighted_features and feature_weights is not None:
        if len(feature_weights) != len(feature_columns):
            raise ValueError("Feature weights must match number of features")
        reference_features = reference_features * feature_weights
        candidate_features = candidate_features * feature_weights

    # Calculate cosine similarities
    reference_reshaped = reference_features.reshape(1, -1).astype(np.float64)
    candidate_features_64 = candidate_features.astype(np.float64)

    cosine_similarities = cosine_similarity(reference_reshaped, candidate_features_64)[
        0
    ]

    # Create ranking with confidence scores
    particle_rankings = []
    for i, (particle_id, similarity) in enumerate(
        zip(particle_ids, cosine_similarities)
    ):
        # Calculate confidence based on similarity gap to second-best
        sorted_sims = np.sort(cosine_similarities)[::-1]  # Descending order
        if len(sorted_sims) > 1:
            confidence = (similarity - sorted_sims[1]) / (
                sorted_sims[0] - sorted_sims[1] + 1e-10
            )
        else:
            confidence = 1.0

        particle_rankings.append({
            "particle_id": particle_id,
            "cosine_similarity": similarity,
            "confidence": confidence,
            "meets_threshold": similarity >= min_similarity_threshold,
            "rank": 0,  # Will be filled below
        })

    # Sort by cosine similarity (descending)
    particle_rankings.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    # Assign ranks
    for rank, particle in enumerate(particle_rankings):
        particle["rank"] = rank + 1

    # Find clear winner vs ambiguous cases
    best_match = particle_rankings[0]
    is_clear_winner = (
        best_match["meets_threshold"]
        and best_match["confidence"] > 0.1  # At least 10% gap to second-best
    )

    return {
        "rankings": particle_rankings,
        "best_match": best_match,
        "is_clear_winner": is_clear_winner,
        "num_above_threshold": sum(
            1 for p in particle_rankings if p["meets_threshold"]
        ),
        "similarity_gap": particle_rankings[0]["cosine_similarity"]
        - (
            particle_rankings[1]["cosine_similarity"]
            if len(particle_rankings) > 1
            else 0
        ),
        "reference_features": reference_features,
        "candidate_features": candidate_features,
    }


def plot_feature_comparisons(
    closest_particles: Dict[str, Any],
    save_path: Path,
    closest_particle_at_t1: int,
    save_pdf: bool = False,
    feature_columns: List[str] = [],
    type_of_features: Literal["pca", "feature"] = "feature",
    comparison_method: Literal[
        "percentage", "cosine", "both", "cosine_ranking"
    ] = "percentage",
    min_similarity_threshold: float = 0.7,
) -> List[int]:
    closest_particles_lst = closest_particles["t1"]
    particles_df = closest_particles_lst["particles"]
    if not feature_columns:
        feature_columns = [
            col for col in particles_df.columns if col.startswith("feature_")
        ]
    features_lst = feature_columns

    # Get t0 features that match the selected feature columns
    if "particles" in closest_particles["t0"]:
        particles_t0 = closest_particles["t0"]["particles"]
        if len(particles_t0) == 0:
            # No particles found at t0 - check if features array has data
            features_t0 = closest_particles["t0"]["features"]
            if len(features_t0) == 0:
                # No particles or features at t0 - return empty list or skip comparison
                print(
                    "Warning: No particles or features found at t0, skipping comparison"
                )
                return []
            particle_t0_features = features_t0[0]
        else:
            particle_t0_features = particles_t0[features_lst].iloc[0].values
    else:
        features_t0 = closest_particles["t0"]["features"]
        if len(features_t0) == 0:
            print("Warning: No features found at t0, skipping comparison")
            return []
        particle_t0_features = features_t0[0]
    particle_ids = particles_df.index.values

    # Extract features for all particles at t1
    particles_t1_features = particles_df[
        features_lst
    ].values  # Shape: (n_particles, n_features)

    # Handle new cosine ranking method
    if comparison_method == "cosine_ranking":
        ranking_results = rank_particles_by_cosine_similarity(
            closest_particles=closest_particles,
            feature_columns=features_lst,
            min_similarity_threshold=min_similarity_threshold,
        )

        if "error" in ranking_results:
            print(f"Error in cosine ranking: {ranking_results['error']}")
            return []

        # Print detailed ranking results
        print(f"\n🏆 COSINE SIMILARITY RANKING RESULTS:")
        print("=" * 60)
        print(f"Similarity threshold: {min_similarity_threshold:.3f}")
        print(f"Particles above threshold: {ranking_results['num_above_threshold']}")
        print(f"Clear winner: {'Yes' if ranking_results['is_clear_winner'] else 'No'}")
        print(f"Similarity gap (1st-2nd): {ranking_results['similarity_gap']:.6f}")

        print("\nRanking (All Candidates):")
        print(
            "   Rank | Particle ID |   Similarity   |  Difference  | Confidence | Status"
        )
        print(
            "   -----|-------------|----------------|--------------|------------|--------"
        )

        best_similarity = ranking_results["rankings"][0]["cosine_similarity"]

        for i, particle in enumerate(ranking_results["rankings"]):
            status_icons = []
            if particle["particle_id"] == closest_particle_at_t1:
                status_icons.append("★ spatial")
            if particle["meets_threshold"]:
                status_icons.append("✓ threshold")
            if particle["rank"] == 1:
                status_icons.append("🥇 best")

            status_str = " ".join(status_icons) if status_icons else ""

            # Calculate difference from best for high-similarity cases
            diff_from_best = best_similarity - particle["cosine_similarity"]

            print(
                f"   {particle['rank']:4d} | {particle['particle_id']:11d} | "
                f"{particle['cosine_similarity']:14.9f} | "
                f"{diff_from_best:12.9f} | "
                f"{particle['confidence']:10.3f} | {status_str}"
            )

        # Show detailed analysis for high-similarity cases
        if best_similarity > 0.98:
            print(f"\n🔍 HIGH SIMILARITY ANALYSIS (all > {0.98:.2f}):")
            print(f"   Smallest difference: {ranking_results['similarity_gap']:.9f}")

            # Find particles that are very close to the best
            close_competitors = [
                p
                for p in ranking_results["rankings"][1:]
                if (best_similarity - p["cosine_similarity"]) < 0.001
            ]

            if close_competitors:
                print(f"   Close competitors (within 0.001): {len(close_competitors)}")
                for comp in close_competitors:
                    print(
                        f"     Particle {comp['particle_id']}: "
                        f"diff = {best_similarity - comp['cosine_similarity']:.9f}"
                    )

            # Check if spatial winner is among top candidates
            spatial_rank = next(
                (
                    p["rank"]
                    for p in ranking_results["rankings"]
                    if p["particle_id"] == closest_particle_at_t1
                ),
                None,
            )
            if spatial_rank:
                print(f"   Spatial winner rank: {spatial_rank}")
                if spatial_rank > 1:
                    spatial_particle = next(
                        p
                        for p in ranking_results["rankings"]
                        if p["particle_id"] == closest_particle_at_t1
                    )
                    print(
                        f"   Spatial similarity: {spatial_particle['cosine_similarity']:.9f}"
                    )
                    print(
                        f"   Gap to best: {best_similarity - spatial_particle['cosine_similarity']:.9f}"
                    )

        # Return the ranking for further analysis
        return [p["particle_id"] for p in ranking_results["rankings"]]

    # Calculate distances based on method
    if comparison_method in ["percentage", "both"]:
        # Calculate percentage differences for all particles and features
        all_percentage_diffs = []
        for i, particle_idx in enumerate(particle_ids):
            particle_features = particles_df[features_lst].iloc[i].values
            # Calculate percentage difference: (ground_truth - t1_features) / ground_truth * 100
            # Handle division by zero by adding small epsilon
            epsilon = 1e-10
            percentage_diffs = (
                (particle_t0_features - particle_features)
                / (particle_t0_features + epsilon)
                * 100
            )
            all_percentage_diffs.append(percentage_diffs)
        all_percentage_diffs = np.array(
            all_percentage_diffs
        )  # Shape: (n_particles, n_features)

    if comparison_method in ["cosine", "both"]:
        # Calculate cosine distances (1 - cosine similarity)
        # Reshape reference features for broadcasting
        reference_features = particle_t0_features.reshape(
            1, -1
        )  # Shape: (1, n_features)

        # Calculate cosine similarities between reference and all particles
        # Use float64 explicitly for better precision
        reference_features_64 = reference_features.astype(np.float64)
        particles_t1_features_64 = particles_t1_features.astype(np.float64)
        cosine_similarities = cosine_similarity(
            reference_features_64, particles_t1_features_64
        )[0]  # Shape: (n_particles,)

        # For per-feature analysis, calculate cosine distance for each feature independently
        per_feature_cosine_distances = []
        for feat_idx in range(len(features_lst)):
            ref_feat = particle_t0_features[feat_idx : feat_idx + 1].reshape(
                1, -1
            )  # Single feature
            particle_feats = particles_t1_features[
                :, feat_idx : feat_idx + 1
            ]  # All particles, single feature

            # Handle zero vectors to avoid division by zero
            ref_norm = np.linalg.norm(ref_feat)
            particle_norms = np.linalg.norm(particle_feats, axis=1)

            if ref_norm == 0:
                # If reference is zero, distance is 1 for non-zero particles, 0 for zero particles
                feat_distances = np.where(particle_norms == 0, 0, 1)
            else:
                # Calculate cosine similarity manually to handle edge cases
                similarities = np.zeros(len(particle_norms))
                for i, (particle_feat, particle_norm) in enumerate(
                    zip(particle_feats, particle_norms)
                ):
                    if particle_norm == 0:
                        similarities[i] = (
                            0  # Zero vector has 0 similarity with non-zero vector
                        )
                    else:
                        similarities[i] = np.dot(ref_feat[0], particle_feat) / (
                            ref_norm * particle_norm
                        )
                feat_distances = 1 - similarities

            per_feature_cosine_distances.append(feat_distances)

        per_feature_cosine_distances = np.array(
            per_feature_cosine_distances
        ).T  # Shape: (n_particles, n_features)

    # Determine winning particles based on chosen method
    if comparison_method == "percentage":
        abs_percentage_diffs = np.abs(all_percentage_diffs)
        min_particle_indices = np.argmin(abs_percentage_diffs, axis=0)
        distance_values = all_percentage_diffs
        distance_label = "Percentage Difference (%)"
    elif comparison_method == "cosine":
        min_particle_indices = np.argmin(per_feature_cosine_distances, axis=0)
        distance_values = per_feature_cosine_distances
        distance_label = "Cosine Distance"
    elif comparison_method == "both":
        # Use percentage differences for determining winners by default, but plot both
        abs_percentage_diffs = np.abs(all_percentage_diffs)
        min_particle_indices = np.argmin(abs_percentage_diffs, axis=0)
        distance_values = all_percentage_diffs
        distance_label = "Percentage Difference (%)"
    else:
        # Fallback case
        return []

    closest_particle_ids = particle_ids[min_particle_indices]

    # Count feature wins for each particle
    frequency_counter = Counter(closest_particle_ids)
    winner_id = int(frequency_counter.most_common(1)[0][0])
    winner_feature_wins = frequency_counter.get(winner_id, 0)

    actual_closest_id = closest_particle_at_t1  # This is the actual closest particle ID
    actual_closest_wins = frequency_counter.get(actual_closest_id, 0)

    print(
        f"\nFeature-based winner: Particle {winner_id} with {winner_feature_wins} wins"
    )
    print(
        f"Actual closest particle: Particle {actual_closest_id} with {actual_closest_wins} wins"
    )
    print(f"Margin: {winner_feature_wins - actual_closest_wins} wins")

    # Create the visualization
    n_features = len(features_lst)
    n_cols = 20
    n_rows = int(np.ceil(n_features / n_cols))

    # Set output filename based on comparison method
    if comparison_method == "cosine":
        output_filename = "feature_cosine_distances_plots.pdf"
    elif comparison_method == "both":
        output_filename = "feature_comparisons_both_methods_plots.pdf"
    else:
        output_filename = "feature_percentage_differences_plots.pdf"

    output_pdf_path = save_path.joinpath(output_filename)
    features_wins_ids = []

    if save_pdf:
        with PdfPages(output_pdf_path) as pdf:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
            axes = axes.flatten()

            for feature_idx in range(n_features):
                ax = axes[feature_idx]
                feature_values = distance_values[:, feature_idx]
                x_positions = np.arange(len(particle_ids))
                winning_particle_id = closest_particle_ids[feature_idx]
                bars = ax.bar(x_positions, feature_values, alpha=0.7, width=0.6)

                for i, (bar, value) in enumerate(zip(bars, feature_values)):
                    particle_id = particle_ids[i]

                    if particle_id == winning_particle_id:
                        bar.set_color("limegreen")
                        bar.set_alpha(0.9)
                        bar.set_edgecolor("darkgreen")
                        bar.set_linewidth(2)
                    elif particle_id == actual_closest_id:
                        # Actual closest particle - gold
                        bar.set_color("gold")
                        bar.set_alpha(0.9)
                    else:
                        # Regular particles - color based on distance metric
                        if comparison_method == "cosine":
                            # For cosine distance, use single color scheme (higher distance = worse)
                            bar.set_color("lightcoral" if value > 0.5 else "lightblue")
                        else:
                            # For percentage differences, color based on positive/negative
                            if value > 0:
                                bar.set_color("lightcoral")  # Positive differences
                            else:
                                bar.set_color("lightblue")  # Negative differences

                # Add reference line
                if comparison_method == "cosine":
                    ax.axhline(
                        y=0,
                        color="red",
                        linestyle="-",
                        alpha=0.8,
                        linewidth=2,
                        label="Perfect Similarity",
                    )
                else:
                    ax.axhline(
                        y=0,
                        color="red",
                        linestyle="-",
                        alpha=0.8,
                        linewidth=2,
                        label="No Difference",
                    )

                if winning_particle_id == actual_closest_id:
                    features_wins_ids.append(feature_idx)
                    ax.text(
                        0.02,
                        0.98,
                        f"Winner: {winning_particle_id} ★",
                        transform=ax.transAxes,
                        fontsize=8,
                        fontweight="bold",
                        verticalalignment="top",
                        color="darkgreen",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                        ),
                    )
                else:
                    ax.text(
                        0.02,
                        0.98,
                        f"Winner: {winning_particle_id}",
                        transform=ax.transAxes,
                        fontsize=8,
                        fontweight="bold",
                        verticalalignment="top",
                        color="darkorange",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7
                        ),
                    )

                ax.set_title(f"Feature {feature_idx}", fontsize=10, fontweight="bold")
                ax.set_xlabel("Particle Index", fontsize=8)
                ax.set_ylabel(distance_label, fontsize=8)

                ax.set_xticks(x_positions)
                ax.set_xticklabels(
                    [str(pid) for pid in particle_ids], fontsize=7, rotation=45
                )
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.tick_params(axis="both", which="major", labelsize=7)

            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)

            # Create legend based on comparison method
            if comparison_method == "cosine":
                legend_elements = [
                    Patch(facecolor="red", alpha=0.8, label="Perfect Similarity (0)"),
                    Patch(facecolor="limegreen", alpha=0.9, label="Feature Winner"),
                    Patch(facecolor="gold", alpha=0.9, label="Actual Closest Particle"),
                    Patch(
                        facecolor="lightcoral", alpha=0.7, label="High Distance (>0.5)"
                    ),
                    Patch(
                        facecolor="lightblue", alpha=0.7, label="Low Distance (≤0.5)"
                    ),
                ]
            else:
                legend_elements = [
                    Patch(facecolor="red", alpha=0.8, label="Ground Truth (0%)"),
                    Patch(facecolor="limegreen", alpha=0.9, label="Feature Winner"),
                    Patch(facecolor="gold", alpha=0.9, label="Actual Closest Particle"),
                    Patch(
                        facecolor="lightcoral", alpha=0.7, label="Positive % Difference"
                    ),
                    Patch(
                        facecolor="lightblue", alpha=0.7, label="Negative % Difference"
                    ),
                ]

            axes[0].legend(handles=legend_elements, fontsize=7, loc="upper right")
            plt.tight_layout()
            plt.subplots_adjust(top=0.94)
            pdf.savefig(fig, bbox_inches="tight", dpi=300)

    # Print analysis results
    frequency_counter = Counter(closest_particle_ids)
    print(f"\nFeature Winner Analysis ({comparison_method} method):")
    print("=" * 60)
    print(f"Total features: {n_features}")
    print(
        f"Feature-based winner: Particle {winner_id} with {winner_feature_wins} wins ({winner_feature_wins / n_features * 100:.1f}%)"
    )
    print(
        f"Actual closest particle: Particle {actual_closest_id} with {actual_closest_wins} wins ({actual_closest_wins / n_features * 100:.1f}%)"
    )
    print(f"Margin: {winner_feature_wins - actual_closest_wins} features")

    # If using cosine distances, also print overall cosine similarities
    if comparison_method in ["cosine", "both"]:
        print("\nOverall cosine similarities:")
        print(
            f"Precision info: Using {cosine_similarities.dtype} with ~{np.finfo(cosine_similarities.dtype).precision} decimal digits"
        )

        # Find the particle with highest overall cosine similarity
        best_overall_idx = np.argmax(cosine_similarities)
        best_overall_particle = particle_ids[best_overall_idx]

        for i, (particle_id, cosine_sim) in enumerate(
            zip(particle_ids, cosine_similarities)
        ):
            markers = []
            if particle_id == actual_closest_id:
                markers.append("★ spatial")
            if particle_id == best_overall_particle:
                markers.append("🏆 feature")
            if particle_id == winner_id:
                markers.append("🎯 per-feat")
            marker_str = " ".join(markers) if markers else ""

            # Show more decimal places to reveal precision issues
            print(f"  Particle {particle_id}: {cosine_sim:.15f} {marker_str}")

        print("\nComparison Summary:")
        print(f"  Spatial closest: {actual_closest_id} (distance-based)")
        print(
            f"  Best overall features: {best_overall_particle} (cosine: {cosine_similarities[best_overall_idx]:.6f})"
        )
        print(
            f"  Per-feature winner: {winner_id} ({winner_feature_wins}/{n_features} wins)"
        )

        # Check for potential precision issues
        min_diff = np.min(np.diff(np.sort(cosine_similarities)))
        if min_diff < 1e-10:
            print(
                f"  WARNING: Very small differences detected (min diff: {min_diff:.2e})"
            )
            print(f"  This may indicate precision issues affecting results.")

    return features_wins_ids
