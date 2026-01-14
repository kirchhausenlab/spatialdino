from collections import defaultdict
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Literal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from skimage import io
import stackview
from matplotlib.axes import Axes


def plot_timepoints(
    file_paths: List[Path],
) -> None:
    n = len(file_paths)
    step = max(n // 6, 1)  # show up to 6 time points
    selected_indices = list(range(0, n, step))[:6]  # ensure we don’t go over 6 columns

    fig, ax = plt.subplots(2, len(selected_indices), figsize=(20, 6))
    if len(selected_indices) == 1:
        ax = ax[:, None]  # ensure 2D indexing if only one column

    for idx, i in enumerate(selected_indices):
        file_path = file_paths[i]
        print(f"Plotting timepoint {i}/{n}")

        volume_file = None
        for name in ["volume.tif", "volume_unnorm.tif"]:
            candidate = file_path.joinpath(name)
            if candidate.exists():
                volume_file = candidate
                break
        if volume_file:
            volume = io.imread(volume_file)
            stackview.imshow(volume, plot=ax[0, idx], labels=False)
            ax[0, idx].set_title(f"Volume {i}")
        else:
            ax[0, idx].set_visible(False)

        seg_file = None
        for name in ["new_instance_seg.tif", "instance_seg.tif"]:
            candidate = file_path.joinpath(name)
            if candidate.exists():
                seg_file = candidate
                break
        if seg_file:
            seg = io.imread(seg_file)
            stackview.imshow(seg, plot=ax[1, idx], labels=True)
            ax[1, idx].set_title(f"Instance Seg {i}")
        else:
            ax[1, idx].set_visible(False)

    fig.tight_layout()
    plt.show()


def visualize_unmatched_particle_3d_trajectories(
    unmatched_particle_id: int,
    exp1_df: pd.DataFrame,
    exp2_df: pd.DataFrame,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    proximity_threshold: float = 10.0,
    min_proximity_threshold: float = 3.0,
    max_nearby_particles: int = 10,
    figsize: Tuple[int, int] = (16, 12),
    alpha_trajectories: float = 0.7,
    line_width: float = 2.0,
    pdf: Optional[PdfPages] = None,
) -> Dict[str, Any]:
    # Get the unmatched particle data
    unmatched_particle = exp1_df[exp1_df["ID"] == unmatched_particle_id].copy()

    if len(unmatched_particle) == 0:
        raise ValueError(f"Particle ID {unmatched_particle_id} not found in exp1_df")

    unmatched_particle = unmatched_particle.sort_values("t")

    # Calculate distances to all edges if volume_shape is provided
    edge_distances = None
    min_edge_distance = None
    closest_edge = None

    if volume_shape is not None:
        z_max, y_max, x_max = volume_shape
        particle_coords = np.array(unmatched_particle[["x", "y", "z"]].values)

        # Calculate distances to all edges
        edge_distances = {
            "x_min": np.min(particle_coords[:, 0]),
            "x_max": x_max - np.max(particle_coords[:, 0]),
            "y_min": np.min(particle_coords[:, 1]),
            "y_max": y_max - np.max(particle_coords[:, 1]),
            "z_min": np.min(particle_coords[:, 2]),
            "z_max": z_max - np.max(particle_coords[:, 2]),
        }

        min_edge_distance = min(edge_distances.values())
        closest_edge = min(edge_distances.keys(), key=lambda x: edge_distances[x])

    # Find all nearby particles from exp2 across all timepoints
    nearby_particles_info = []

    for _, unmatched_point in unmatched_particle.iterrows():
        t = unmatched_point["t"]
        unmatched_pos = np.array([
            unmatched_point["x"],
            unmatched_point["y"],
            unmatched_point["z"],
        ])

        # Get all exp2 particles at this timepoint
        exp2_at_t = exp2_df[exp2_df["t"] == t]

        for _, exp2_point in exp2_at_t.iterrows():
            exp2_pos = np.array([exp2_point["x"], exp2_point["y"], exp2_point["z"]])
            distance = np.linalg.norm(unmatched_pos - exp2_pos)

            if min_proximity_threshold <= distance <= proximity_threshold:
                nearby_particles_info.append({
                    "exp2_id": exp2_point["ID"],
                    "distance": distance,
                    "t": t,
                    "exp2_x": exp2_point["x"],
                    "exp2_y": exp2_point["y"],
                    "exp2_z": exp2_point["z"],
                    "unmatched_x": unmatched_point["x"],
                    "unmatched_y": unmatched_point["y"],
                    "unmatched_z": unmatched_point["z"],
                })

    if not nearby_particles_info:
        return {"error": "No nearby particles found"}

    # Convert to DataFrame and analyze
    nearby_df = pd.DataFrame(nearby_particles_info)

    # Get unique exp2 particle IDs and their average distances
    exp2_particle_stats = (
        nearby_df.groupby("exp2_id")
        .agg({"distance": ["mean", "min", "max", "count"], "t": ["min", "max"]})
        .round(2)
    )

    exp2_particle_stats.columns = [
        "mean_dist",
        "min_dist",
        "max_dist",
        "encounters",
        "t_min",
        "t_max",
    ]
    exp2_particle_stats = exp2_particle_stats.sort_values("mean_dist").head(
        max_nearby_particles
    )

    # Get full trajectories for the selected nearby particles
    nearby_particle_ids = exp2_particle_stats.index.tolist()
    nearby_trajectories = {}

    for particle_id in nearby_particle_ids:
        trajectory = exp2_df[exp2_df["ID"] == particle_id].copy().sort_values("t")
        nearby_trajectories[particle_id] = trajectory

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # 3D plot (left side)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.view_init(elev=20, azim=45)  # type: ignore

    # Color setup
    colormap = plt.cm.get_cmap("tab10")
    colors = [
        colormap(i / len(nearby_particle_ids)) for i in range(len(nearby_particle_ids))
    ]

    # Plot unmatched particle with distinct styling
    exp1_color = "#FF6B35"  # Distinct orange-red for Exp1

    ax1.plot(
        unmatched_particle["x"],
        unmatched_particle["y"],
        unmatched_particle["z"],
        color=exp1_color,
        linewidth=line_width * 2.5,
        alpha=alpha_trajectories,
        label=f"Exp1 ID {unmatched_particle_id} (Unmatched)",
    )

    # Start/end markers for unmatched particle
    ax1.scatter(
        *unmatched_particle.iloc[0][["x", "y", "z"]],
        color="green",
        s=150,
        marker="o",
        label="Start",
        edgecolors="black",
        linewidth=2,
    )
    ax1.scatter(
        *unmatched_particle.iloc[-1][["x", "y", "z"]],
        color="red",
        s=150,
        marker="s",
        label="End",
        edgecolors="black",
        linewidth=2,
    )

    # Add ID label for unmatched particle at midpoint
    mid_idx = len(unmatched_particle) // 2
    mid_point = unmatched_particle.iloc[mid_idx]
    ax1.text(
        mid_point["x"],
        mid_point["y"],
        mid_point["z"],
        f"ID{unmatched_particle_id}",  # type: ignore
        fontsize=12,
        fontweight="bold",
        color=exp1_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Plot nearby trajectories
    for i, (pid, traj) in enumerate(nearby_trajectories.items()):
        stats = exp2_particle_stats.loc[pid]
        ax1.plot(
            traj["x"],
            traj["y"],
            traj["z"],
            color=colors[i],
            linewidth=line_width,
            alpha=alpha_trajectories,
            label=f"Exp2 ID {pid} (avg:{stats['mean_dist']:.1f}px)",
        )
        # Start/end markers for nearby particles
        ax1.scatter(
            *traj.iloc[0][["x", "y", "z"]],
            color=colors[i],
            s=80,
            marker="o",
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
        ax1.scatter(
            *traj.iloc[-1][["x", "y", "z"]],
            color=colors[i],
            s=80,
            marker="s",
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add ID labels for nearby particles
        mid_idx = len(traj) // 2
        mid_point = traj.iloc[mid_idx]
        ax1.text(
            mid_point["x"],
            mid_point["y"],
            mid_point["z"],
            f"ID{pid}",  # type: ignore
            fontsize=10,
            color=colors[i],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    # 3D plot styling
    ax1.set_xlabel("X (px)")
    ax1.set_ylabel("Y (px)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Enhanced title with key statistics
    dists = nearby_df["distance"].values
    title = f"3D Trajectories: Exp1 ID {unmatched_particle_id} + {len(nearby_particle_ids)} nearby particles"
    if edge_distances:
        title += f"\nClosest edge: {min_edge_distance:.1f}px | Avg distance: {dists.mean():.1f}px"  # type: ignore

    ax1.set_title(title, fontsize=10, pad=20)

    # Time series plot (right side)
    ax2 = fig.add_subplot(1, 2, 2)

    # Plot Exp1 particle trajectory over time (using distance from origin as a metric)
    exp1_distances = np.sqrt(
        unmatched_particle["x"] ** 2
        + unmatched_particle["y"] ** 2
        + unmatched_particle["z"] ** 2
    )

    ax2.plot(
        unmatched_particle["t"],
        exp1_distances,
        color=exp1_color,
        linewidth=3,
        marker="o",
        markersize=6,
        label=f"Exp1 ID {unmatched_particle_id}",
        alpha=0.9,
    )

    # Plot nearby particles trajectories over time
    for i, (pid, traj) in enumerate(nearby_trajectories.items()):
        exp2_distances = np.sqrt(traj["x"] ** 2 + traj["y"] ** 2 + traj["z"] ** 2)

        ax2.plot(
            traj["t"],
            exp2_distances,
            color=colors[i],
            linewidth=2,
            marker="s",
            markersize=4,
            label=f"Exp2 ID {pid}",
            alpha=0.7,
        )

    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Distance from Origin (px)")
    ax2.set_title("Particle Trajectories Over Time")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add frame range annotation
    frame_range = (
        f"Frames: {unmatched_particle['t'].min()}-{unmatched_particle['t'].max()}"
    )
    ax2.text(
        0.02,
        0.98,
        frame_range,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "unmatched_particle_id": unmatched_particle_id,
        "unmatched_trajectory": unmatched_particle,
        "nearby_particles": nearby_trajectories,
        "nearby_particle_stats": exp2_particle_stats,
        "encounter_data": nearby_df,
        "edge_distances": edge_distances,
        "min_edge_distance": min_edge_distance,
        "closest_edge": closest_edge,
        "distance_stats": {
            "min": dists.min(),  # type: ignore
            "max": dists.max(),  # type: ignore
            "mean": dists.mean(),  # type: ignore
            "median": np.median(dists),  # type: ignore
            "std": np.std(dists),  # type: ignore
        },
    }


def analyze_multiple_unmatched_particles_3d(
    unmatched_particle_ids: List[int],
    exp1_df: pd.DataFrame,
    exp2_df: pd.DataFrame,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    proximity_threshold: float = 10.0,
    min_proximity_threshold: float = 3.0,
    max_nearby_particles: int = 5,
    i_start: int = 0,
    i_end: int = 10,
    output_pdf_path: str = "unmatched_analysis_plots.pdf",
) -> Dict[int, Dict[str, Any]]:
    results = {}
    lst_to_show = list(set(sorted(unmatched_particle_ids)))
    print(f"total number of unmatched particles: {len(lst_to_show)}")
    new_lst = []
    for k in range(i_start, i_end):
        new_lst.append(lst_to_show[k])
    print(f"lst to show: {new_lst}")

    with PdfPages(output_pdf_path) as pdf:
        for i, pid in enumerate(tqdm(new_lst, desc="Analyzing particles")):
            print(f"Processing particle {i + 1}/{len(new_lst)}: ID {pid}")
            try:
                res = visualize_unmatched_particle_3d_trajectories(
                    unmatched_particle_id=pid,
                    exp1_df=exp1_df,
                    exp2_df=exp2_df,
                    volume_shape=volume_shape,
                    proximity_threshold=proximity_threshold,
                    min_proximity_threshold=min_proximity_threshold,
                    max_nearby_particles=max_nearby_particles,
                    figsize=(16, 10),
                    pdf=pdf,
                )
                results[pid] = res
            except Exception as e:
                print(f"Error processing particle {pid}: {str(e)}")
                results[pid] = {"error": str(e)}

    print(
        f"Successfully processed {len([r for r in results.values() if 'error' not in r])} particles"
    )
    print(
        f"Errors encountered: {len([r for r in results.values() if 'error' in r])} particles"
    )

    return results


def create_cme_crop_with_dino_overlay(
    img: np.ndarray,
    exp1_coords: Tuple[float, float, float],
    exp2_coords_list: List[Tuple[float, float, float]],
    crop_size: int = 30,
) -> Tuple[np.ndarray, List[Tuple[float, float, float]], Tuple[int, int, int, int]]:
    exp1_x, exp1_y, exp1_z = exp1_coords

    # Define crop boundaries
    half_size = crop_size // 2
    x_min = max(0, int(exp1_x - half_size))
    x_max = min(img.shape[2], int(exp1_x + half_size))
    y_min = max(0, int(exp1_y - half_size))
    y_max = min(img.shape[1], int(exp1_y + half_size))

    # Extract crop from all z slices
    crop_3d = img[:, y_min:y_max, x_min:x_max]

    # Create maximum projection
    max_proj = np.max(crop_3d, axis=0)

    # Find dino particles within crop region
    exp2_points_in_crop = []
    for exp2_x, exp2_y, exp2_z in exp2_coords_list:
        if x_min <= exp2_x < x_max and y_min <= exp2_y < y_max:
            crop_x = exp2_x - x_min
            crop_y = exp2_y - y_min
            exp2_points_in_crop.append((crop_x, crop_y, exp2_z))

    crop_bounds = (x_min, x_max, y_min, y_max)

    return max_proj, exp2_points_in_crop, crop_bounds


def visualize_unmatched_crops(
    img: np.ndarray,
    unmatched_exp1_particles: List[Dict[str, Any]],
    exp2_coords_list: List[Tuple[float, float, float]],
    max_crops_to_show: int = 20,
    crop_size: int = 30,
    figsize_per_crop: int = 10,
    n_closest_unmatched: int = 10,
    output_pdf_path: str = "unmatched_crops.pdf",
) -> None:
    n_crops = min(len(unmatched_exp1_particles), max_crops_to_show)

    # Calculate grid dimensions
    cols = min(5, n_crops)
    rows = (n_crops + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per_crop, rows * figsize_per_crop)
    )
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    print(
        f"\nGenerating {n_crops} crops from {len(unmatched_exp1_particles)} unmatched exp1 particles..."
    )
    with PdfPages(output_pdf_path) as pdf:
        for i in tqdm(range(n_crops), total=n_crops):
            exp_particle = unmatched_exp1_particles[i]
            exp_coords = exp_particle["coordinates"]
            exp_id = exp_particle["exp1_id"]

            print(f"\n--- CROP {i + 1}/{n_crops} ---")
            print(
                f"Processing exp1 ID: {int(exp_id)}, Coordinates: ({exp_coords[0]:.2f}, {exp_coords[1]:.2f}, {exp_coords[2]:.2f})"
            )

            # Generate crop and find dino particles
            max_proj, exp2_points_in_crop, bounds = create_cme_crop_with_dino_overlay(
                img, exp_coords, exp2_coords_list, crop_size=crop_size
            )

            print(f"Found {len(exp2_points_in_crop)} DINO particles in this crop:")
            for j, (exp2_x, exp2_y, exp2_z) in enumerate(exp2_points_in_crop):
                orig_exp2_x = exp2_x + bounds[0]
                orig_exp2_y = exp2_y + bounds[2]
                distance_3d = np.sqrt(
                    (exp_coords[0] - orig_exp2_x) ** 2
                    + (exp_coords[1] - orig_exp2_y) ** 2
                    + (exp_coords[2] - exp2_z) ** 2
                )

            # Plot the maximum projection
            ax = axes[i] if n_crops > 1 else axes[0]
            im = ax.imshow(max_proj, cmap="gray", interpolation="nearest")

            # Convert CME coordinates to crop coordinates
            exp_center_x = exp_coords[0] - bounds[0]
            exp_center_y = exp_coords[1] - bounds[2]

            # Calculate distances and create list of (distance, index, coordinates) tuples
            distance_data = []
            for j, (exp2_x, exp2_y, exp2_z) in enumerate(exp2_points_in_crop):
                orig_exp2_x = exp2_x + bounds[0]
                orig_exp2_y = exp2_y + bounds[2]

                distance_3d = np.sqrt(
                    (exp_coords[0] - orig_exp2_x) ** 2
                    + (exp_coords[1] - orig_exp2_y) ** 2
                    + (exp_coords[2] - exp2_z) ** 2
                )

                distance_data.append((
                    distance_3d,
                    j,
                    exp2_x,
                    exp2_y,
                    exp2_z,
                    orig_exp2_x,
                    orig_exp2_y,
                ))

            # Sort by distance and get top 5 closest
            distance_data.sort(key=lambda x: x[0])
            top_5_closest = distance_data[:n_closest_unmatched]

            distances = []
            for rank, (
                distance_3d,
                j,
                exp2_x,
                exp2_y,
                exp2_z,
                orig_exp2_x,
                orig_exp2_y,
            ) in enumerate(top_5_closest):
                distances.append(distance_3d)

                # Use color based on rank
                color = "#FF0000"

                # Enhanced marker for top 5
                ax.plot(
                    exp2_x, exp2_y, "r+", markersize=14, markeredgewidth=3, alpha=1.0
                )

                # Draw distance line
                ax.plot(
                    [exp_center_x, exp2_x],
                    [exp_center_y, exp2_y],
                    linestyle="--",
                    linewidth=4,
                    color=color,
                    alpha=0.8,
                )

                # Calculate annotation position with offset to prevent overlap
                mid_x, mid_y = (exp_center_x + exp2_x) / 2, (exp_center_y + exp2_y) / 2

                # Add small offset based on rank to prevent overlap
                offset_x = (rank - 2) * 5  # Spread annotations horizontally
                offset_y = (rank - 2) * 5  # Spread annotations vertically

                ax.annotate(
                    f"#{rank + 1}: {distance_3d:.1f}",
                    (mid_x, mid_y),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    fontsize=16,
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

            # Plot CME particle
            ax.plot(
                exp_center_x,
                exp_center_y,
                "bo",
                markersize=16,
                markerfacecolor="none",
                markeredgewidth=4,
                alpha=0.8,
            )
            ax.annotate(
                "exp1",
                (exp_center_x, exp_center_y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=18,
                color="blue",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

            if distances:
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = np.mean(distances)
                title_text = f"exp1 ID{int(exp_id)}: ({exp_coords[0]:.0f},{exp_coords[1]:.0f},{exp_coords[2]:.0f})\n{len(exp2_points_in_crop)} DINO pts | Top {n_closest_unmatched} closest: {min_dist:.1f}-{max_dist:.1f} | Avg: {avg_dist:.1f}"
            else:
                title_text = f"exp1 ID{int(exp_id)}: ({exp_coords[0]:.0f},{exp_coords[1]:.0f},{exp_coords[2]:.0f})\n{len(exp2_points_in_crop)} DINO pts"

            ax.set_title(title_text, fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n_crops, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        # plt.close(fig)


def plot_track_lengths(
    linked: pd.DataFrame,
    ax: Axes,
    experiment_name: str,
) -> pd.DataFrame:
    # track id can be either ID or track_id
    if "track_id" in linked.columns:
        track_id_column = "track_id"
    else:
        track_id_column = "ID"

    track_lengths = linked["track_length"].groupby(linked[track_id_column]).first()
    unique_track_ids = track_lengths.index.tolist()
    print(f"Total number of unique tracks: {len(unique_track_ids)}")
    length_counts = track_lengths.value_counts().sort_index(ascending=False)

    summary_df = pd.DataFrame({
        "track_length": length_counts.index,
        "number_of_tracks": length_counts.values,
    }).sort_values("track_length", ascending=True)

    ax.bar(length_counts.index, length_counts.values, edgecolor="black", alpha=0.7)  # type: ignore

    for length, count in length_counts.items():
        ax.text(length, count + 0.1, str(count), ha="center", va="bottom", fontsize=9)  # type: ignore

    ax.set_xlabel("Track Length (frames)")
    ax.set_ylabel("Number of Tracks")
    ax.set_title(f"Distribution of Track Lengths - {experiment_name}")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    return summary_df


def plot_track_lengths_for_multiple_experiments(
    linked_dfs: List[pd.DataFrame],
    experiment_name: str,
) -> None:
    fig, axs = plt.subplots(1, len(linked_dfs), figsize=(25, 15))
    for i, linked_df in enumerate(linked_dfs):
        ax = axs[i]
        # track id can be either ID or track_id
        if "track_id" in linked_df.columns:
            track_id_column = "track_id"
        else:
            track_id_column = "ID"

        track_lengths = (
            linked_df["track_length"].groupby(linked_df[track_id_column]).first()
        )
        unique_track_ids = track_lengths.index.tolist()
        print(f"Total number of unique tracks: {len(unique_track_ids)}")

        # get the top 10 longest tracks
        top_10_tracks = track_lengths.sort_values(ascending=False).head(10)
        print(f"Top 10 longest tracks have frequency: {top_10_tracks.value_counts()}")

        length_counts = track_lengths.value_counts().sort_index(ascending=False)
        # Create the bar plot
        bars = ax.bar(
            length_counts.index,  # type: ignore
            length_counts.values,  # type: ignore
            edgecolor="black",
            alpha=0.7,  # type: ignore
        )

        # Add particle counts on top of bars
        max_height = max(length_counts.values) if len(length_counts.values) > 0 else 1
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                x_center = bar.get_x() + bar.get_width() / 2.0
                ax.text(
                    x_center,
                    height + max_height * 0.02,  # Use 2% of max height as offset
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )

        # Adjust y-axis to show labels
        ax.set_ylim(0, max_height * 1.1)  # Add 10% padding at top

        ax.set_xlabel("Track Length (frames)")
        ax.set_ylabel("Number of Tracks")
        ax.set_title(f"Distribution of Track Lengths - {experiment_name}")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    plt.show()


# Create new figure for top 10 longest tracks side-by-side comparison
def _get_top_tracks_by_length(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    track_id_column = "track_id" if "track_id" in df.columns else "ID"
    track_lengths = df["track_length"].groupby(df[track_id_column]).first()
    return track_lengths.value_counts().sort_index(ascending=False).head(top_n)


def _get_track_length_data(df: pd.DataFrame) -> pd.Series:
    track_id_column = "track_id" if "track_id" in df.columns else "ID"
    track_lengths = df["track_length"].groupby(df[track_id_column]).first()
    return track_lengths.value_counts().sort_index()


def plot_particles_per_timepoint(
    combined_data: pd.DataFrame,
    track_df_1: pd.DataFrame,
    track_df_2: pd.DataFrame,
    track_df_3: pd.DataFrame,
    experiment_1_name: str = "Experiment 1",
    experiment_2_name: str = "Experiment 2",
    experiment_3_name: str = "Experiment 3",
) -> None:
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Row 0: Particles per timepoint for each channel
    channels = list(combined_data["channel"].unique())
    for i, (channel, data) in enumerate(combined_data.groupby("channel")):
        if i < 3:  # Only plot first 3 channels
            ax = axes[0, i]
            ax.plot(data["t"], data["particle_count"], "o-", linewidth=2, markersize=4)

            # Add count labels on top of each point
            for _, row in data.iterrows():
                ax.text(
                    row["t"],
                    row["particle_count"] + 2,
                    str(row["particle_count"]),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax.set_xlabel("Time Point")
            ax.set_ylabel("Number of Particles")
            ax.set_title(f"{channel} Particles per Timepoint")
            ax.grid(True, alpha=0.3)

    for i in range(len(channels), 3):
        axes[0, i].set_visible(False)

    track_dfs = [track_df_1, track_df_2, track_df_3]
    experiment_names = [experiment_1_name, experiment_2_name, experiment_3_name]

    for i, (track_df, exp_name) in enumerate(zip(track_dfs, experiment_names)):
        ax = axes[1, i]
        plot_track_lengths(track_df, ax=ax, experiment_name=exp_name)

    # Row 2, Plot 0: Combined particles per timepoint for all channels
    ax = axes[2, 0]
    for channel, data in combined_data.groupby("channel"):
        ax.plot(
            data["t"],
            data["particle_count"],
            "o-",
            label=channel,
            linewidth=2,
            markersize=4,
        )

        # Add count labels on top of each point for combined plot
        for _, row in data.iterrows():
            ax.text(
                row["t"],
                row["particle_count"] + 2,
                str(row["particle_count"]),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel("Time Point")
    ax.set_ylabel("Number of Particles")
    ax.set_title("Particles per Timepoint - All Channels")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2, Plot 1: Track length comparison between experiment 1 and 2
    ax = axes[2, 1]
    length_counts_1 = _get_track_length_data(track_df_1)
    length_counts_2 = _get_track_length_data(track_df_2)
    all_lengths_12 = set(length_counts_1.index) | set(length_counts_2.index)

    ax.bar(
        length_counts_1.index,
        length_counts_1.values,
        alpha=0.7,
        label=experiment_1_name,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        length_counts_2.index,
        length_counts_2.values,
        alpha=0.7,
        label=experiment_2_name,
        edgecolor="black",
        linewidth=0.5,
    )

    min_length_12, max_length_12 = min(all_lengths_12), max(all_lengths_12)
    ax.set_xlabel("Track Length (frames)")
    ax.set_ylabel("Number of Tracks")
    ax.set_title(f"Track Length: {experiment_1_name} vs {experiment_2_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_length_12 - 0.5, max_length_12 + 0.5)
    ax.invert_xaxis()

    # Row 2, Plot 2: Track length comparison between experiment 1 and 3
    ax = axes[2, 2]
    length_counts_3 = _get_track_length_data(track_df_3)
    all_lengths_13 = set(length_counts_1.index) | set(length_counts_3.index)

    ax.bar(
        length_counts_1.index,
        length_counts_1.values,
        alpha=0.7,
        label=experiment_1_name,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        length_counts_3.index,
        length_counts_3.values,
        alpha=0.7,
        label=experiment_3_name,
        edgecolor="black",
        linewidth=0.5,
    )

    min_length_13, max_length_13 = min(all_lengths_13), max(all_lengths_13)
    ax.set_xlabel("Track Length (frames)")
    ax.set_ylabel("Number of Tracks")
    ax.set_title(f"Track Length: {experiment_1_name} vs {experiment_3_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_length_13 - 0.5, max_length_13 + 0.5)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    # Create separate figure for top 10 longest tracks side-by-side comparison for all 3 experiments
    top_tracks_1 = _get_top_tracks_by_length(track_df_1)
    top_tracks_2 = _get_top_tracks_by_length(track_df_2)
    top_tracks_3 = _get_top_tracks_by_length(track_df_3)

    all_track_lengths = sorted(
        set(top_tracks_1.index) | set(top_tracks_2.index) | set(top_tracks_3.index),
        reverse=True,
    )

    counts_1 = [top_tracks_1.get(length, 0) for length in all_track_lengths]
    counts_2 = [top_tracks_2.get(length, 0) for length in all_track_lengths]
    counts_3 = [top_tracks_3.get(length, 0) for length in all_track_lengths]

    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))
    x = np.arange(len(all_track_lengths))
    width = 0.25

    bars1 = ax2.bar(
        x - width,
        counts_1,
        width,
        label=experiment_1_name,
        edgecolor="black",
        alpha=0.7,
        color="skyblue",
    )
    bars2 = ax2.bar(
        x,
        counts_2,
        width,
        label=experiment_2_name,
        edgecolor="black",
        alpha=0.7,
        color="lightcoral",
    )
    bars3 = ax2.bar(
        x + width,
        counts_3,
        width,
        label=experiment_3_name,
        edgecolor="black",
        alpha=0.7,
        color="lightgreen",
    )

    ax2.set_xlabel("Track Length (frames)")
    ax2.set_ylabel("Number of Tracks")
    ax2.set_title("Top 10 Longest Track Lengths - Three-Way Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(length) for length in all_track_lengths])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show labels for non-zero bars
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # invert x axis
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.show()


def plot_particles_per_timepoint_for_2_experiments(
    combined_data: pd.DataFrame,
    track_df_1: pd.DataFrame,
    track_df_2: pd.DataFrame,
    experiment_1_name: str,
    experiment_2_name: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (channel, data) in enumerate(combined_data.groupby("channel")):
        ax = axes[0, i]
        ax.plot(data["t"], data["particle_count"], "o-", linewidth=2, markersize=4)

        # Add count labels on top of each point
        for _, row in data.iterrows():
            ax.text(
                row["t"],
                row["particle_count"] + 2,
                str(row["particle_count"]),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel("Time Point")
        ax.set_ylabel("Number of Particles")
        ax.set_title(f"{channel} Particles per Timepoint")
        ax.grid(True, alpha=0.3)

    # Combined comparison plot
    ax = axes[1, 2]
    for channel, data in combined_data.groupby("channel"):
        ax.plot(
            data["t"],
            data["particle_count"],
            "o-",
            label=channel,
            linewidth=2,
            markersize=4,
        )

        # Add count labels on top of each point for combined plot
        for _, row in data.iterrows():
            ax.text(
                row["t"],
                row["particle_count"] + 2,
                str(row["particle_count"]),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel("Time Point")
    ax.set_ylabel("Number of Particles")
    ax.set_title("Particles per Timepoint - Both Channels")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Track length comparison plot
    ax = axes[1, 0]
    plot_track_lengths(track_df_1, ax=ax, experiment_name=experiment_1_name)
    ax = axes[1, 1]
    plot_track_lengths(track_df_2, ax=ax, experiment_name=experiment_2_name)

    ax = axes[0, 2]

    length_counts_1 = _get_track_length_data(track_df_1)
    length_counts_2 = _get_track_length_data(track_df_2)

    # Get combined range for consistent x-axis
    all_lengths = set(length_counts_1.index) | set(length_counts_2.index)
    min_length, max_length = min(all_lengths), max(all_lengths)

    # Create overlaid bar charts with transparency
    ax.bar(
        length_counts_1.index,
        length_counts_1.values,
        alpha=0.7,
        label=experiment_1_name,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        length_counts_2.index,
        length_counts_2.values,
        alpha=0.7,
        label=experiment_2_name,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Track Length (frames)")
    ax.set_ylabel("Number of Tracks")
    ax.set_title("Track Length Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_length - 0.5, max_length + 0.5)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    # Create new figure for top 10 longest tracks side-by-side comparison
    top_tracks_1 = _get_top_tracks_by_length(track_df_1)
    top_tracks_2 = _get_top_tracks_by_length(track_df_2)
    all_track_lengths = sorted(
        set(top_tracks_1.index) | set(top_tracks_2.index), reverse=True
    )

    counts_1 = [top_tracks_1.get(length, 0) for length in all_track_lengths]
    counts_2 = [top_tracks_2.get(length, 0) for length in all_track_lengths]

    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))
    x = np.arange(len(all_track_lengths))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        counts_1,
        width,
        label=experiment_1_name,
        edgecolor="black",
        alpha=0.7,
        color="skyblue",
    )
    bars2 = ax2.bar(
        x + width / 2,
        counts_2,
        width,
        label=experiment_2_name,
        edgecolor="black",
        alpha=0.7,
        color="lightcoral",
    )

    ax2.set_xlabel("Track Length (frames)")
    ax2.set_ylabel("Number of Tracks")
    ax2.set_title("Top 10 Longest Track Lengths - Side-by-Side Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(length) for length in all_track_lengths])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show labels for non-zero bars
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # invert x axis
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.show()


def calculate_track_length_matches(
    track_df_1: pd.DataFrame,
    track_df_2: pd.DataFrame,
    tolerance: float = 3.5,
    bin_size: int = 10,
) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
    """
    Calculate match ratios between two track datasets based on track lengths.

    Args:
        track_df_1: First track dataset
        track_df_2: Second track dataset
        tolerance: Maximum difference in track length for a match (default 3.5)
        bin_size: Size of bins for grouping track lengths (default 10)

    Returns:
        Tuple of (match_ratios, total_counts_1, matched_counts)
    """
    # Get track length data for both datasets
    length_counts_1 = _get_track_length_data(track_df_1)
    length_counts_2 = _get_track_length_data(track_df_2)

    # Create dictionaries for easier lookup
    lengths_1 = dict(length_counts_1)
    lengths_2 = dict(length_counts_2)

    # Initialize bins
    max_length = max(
        max(lengths_1.keys()) if lengths_1 else [0],
        max(lengths_2.keys()) if lengths_2 else [0],
    )
    bins = list(range(0, max_length + bin_size, bin_size))  # type: ignore

    # Initialize counters for each bin
    bin_total_counts = defaultdict(int)
    bin_matched_counts = defaultdict(int)

    # For each track length in dataset 1, try to find matches in dataset 2
    for length_1, count_1 in lengths_1.items():
        bin_idx = length_1 // bin_size
        bin_total_counts[bin_idx] += count_1  # type: ignore

        # Find matches within tolerance
        matches_found = 0
        for length_2, count_2 in lengths_2.items():
            if abs(length_1 - length_2) <= tolerance:
                # We found a match - count minimum of available tracks
                matches_found += min(count_1, count_2)  # type: ignore
                break

        bin_matched_counts[bin_idx] += matches_found

    # Calculate match ratios for each bin
    match_ratios = {}
    total_counts = {}
    matched_counts = {}

    for bin_idx in bin_total_counts.keys():
        total = bin_total_counts[bin_idx]
        matched = bin_matched_counts[bin_idx]
        match_ratios[bin_idx] = (matched / total * 100) if total > 0 else 0
        total_counts[bin_idx] = total
        matched_counts[bin_idx] = matched

    return match_ratios, total_counts, matched_counts


def plot_track_length_match_histogram(
    track_df_1: pd.DataFrame,
    track_df_2: pd.DataFrame,
    experiment_1_name: str,
    experiment_2_name: str,
    tolerance: float = 3.5,
    bin_size: int = 10,
) -> None:
    """
    Create histogram showing track length match ratios between two experiments.
    """
    # Calculate matches
    match_ratios, total_counts, matched_counts = calculate_track_length_matches(
        track_df_1, track_df_2, tolerance, bin_size
    )

    if not match_ratios:
        print("No data available for matching analysis.")
        return

    # Prepare data for plotting
    bin_labels = []
    ratios = []
    totals = []
    matches = []

    max_bin = max(match_ratios.keys())
    for bin_idx in range(max_bin + 1):
        start_length = bin_idx * bin_size
        end_length = start_length + bin_size - 1
        bin_labels.append(f"{start_length}-{end_length}")
        ratios.append(match_ratios.get(bin_idx, 0))
        totals.append(total_counts.get(bin_idx, 0))
        matches.append(matched_counts.get(bin_idx, 0))

    # Create the histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Match ratios
    bars1 = ax1.bar(
        range(len(bin_labels)),
        ratios,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("Track Length Bins (frames)")
    ax1.set_ylabel("Match Ratio (%)")
    ax1.set_title(
        f"Track Length Match Ratios\n({experiment_1_name} vs {experiment_2_name}, tolerance=±{tolerance} frames)"
    )
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Add percentage labels on bars
    for i, (bar, ratio) in enumerate(zip(bars1, ratios)):
        if ratio > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{ratio:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Bottom plot: Absolute counts (stacked)
    bars2_total = ax2.bar(
        range(len(bin_labels)),
        totals,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
        linewidth=0.5,
        label=f"Total tracks ({experiment_1_name})",
    )
    bars2_matched = ax2.bar(
        range(len(bin_labels)),
        matches,
        alpha=0.9,
        color="darkgreen",
        edgecolor="black",
        linewidth=0.5,
        label="Matched tracks",
    )

    ax2.set_xlabel("Track Length Bins (frames)")
    ax2.set_ylabel("Number of Tracks")
    ax2.set_title("Track Counts: Total vs Matched")
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add count labels on bars
    for i, (total, matched) in enumerate(zip(totals, matches)):
        if total > 0:
            ax2.text(i, total + 0.5, str(total), ha="center", va="bottom", fontsize=8)
        if matched > 0:
            ax2.text(
                i,
                matched + 0.5,
                str(matched),
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                weight="bold",
            )

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_tracks_1 = sum(totals)
    total_matched = sum(matches)
    overall_match_ratio = (
        (total_matched / total_tracks_1 * 100) if total_tracks_1 > 0 else 0
    )

    print("\nSummary Statistics:")
    print(f"Total tracks in {experiment_1_name}: {total_tracks_1}")
    print(f"Total matched tracks: {total_matched}")
    print(f"Overall match ratio: {overall_match_ratio:.2f}%")
    print(f"Tolerance used: ±{tolerance} frames")
    print(f"Bin size: {bin_size} frames")
