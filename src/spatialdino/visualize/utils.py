import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Tuple, List
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm
from skimage import io
from scipy.spatial.distance import cdist
import pandas as pd
from typing import Dict, Any
import stackview
from matplotlib.patches import Rectangle


def interpolate_feature_to_full_size(
    feature_map: np.ndarray,
    target_shape: Tuple[int, int, int] = (60, 608, 546),
) -> np.ndarray:
    feature_tensor = torch.from_numpy(feature_map).unsqueeze(0).unsqueeze(0).float()
    interpolated = F.interpolate(
        feature_tensor, size=target_shape, mode="trilinear", align_corners=False
    )
    return interpolated.squeeze(0).squeeze(0).numpy()


def _get_z_range(
    feature_map: np.ndarray,
    z_range_name: str,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    middle_z = target_shape[0] // 2
    feature_map_obj = feature_map
    if z_range_name == "single_plane":
        feature_map_obj = feature_map[middle_z : middle_z + 1, :, :]
    elif z_range_name == "5_planes":
        start_z = max(0, middle_z - 2)
        end_z = min(target_shape[0], start_z + 5)
        feature_map_obj = feature_map[start_z:end_z, :, :]
    elif z_range_name == "20_planes":
        start_z = max(0, middle_z - 10)
        end_z = min(target_shape[0], start_z + 20)
        feature_map_obj = feature_map[start_z:end_z, :, :]
    else:
        feature_map_obj = feature_map
    return np.max(feature_map_obj, axis=0)


def _setup_figure_object(
    num_pages: int,
    grid_cols: int,
    grid_rows: int,
    start_feat: int,
    end_feat: int,
    z_range_name: str,
) -> Tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(
        grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3)
    )
    fig.suptitle(
        f"Features {start_feat}-{end_feat - 1} - {z_range_name}",
        fontsize=16,
        color="white",
    )
    return fig, axes


def _setup_plotting(
    num_features: int,
    features_per_page: int,
) -> Tuple[int, int, int]:
    num_pages = (num_features + features_per_page - 1) // features_per_page

    # Calculate grid dimensions (try to make it roughly square)
    grid_cols = int(np.ceil(np.sqrt(features_per_page)))
    grid_rows = int(np.ceil(features_per_page / grid_cols))
    return num_pages, grid_cols, grid_rows


def create_feature_summary_grid(
    lr_feats: torch.Tensor,
    target_shape: Tuple[int, int, int] = (60, 608, 546),
    features_per_page: int = 64,
    z_range_name: str = "full",
) -> None:
    _, _, _, num_features = lr_feats.shape
    num_pages, grid_cols, grid_rows = _setup_plotting(num_features, features_per_page)

    for page_idx in range(num_pages):
        start_feat = page_idx * features_per_page
        end_feat = min((page_idx + 1) * features_per_page, num_features)

        fig, axes = _setup_figure_object(
            num_pages, grid_cols, grid_rows, start_feat, end_feat, z_range_name
        )
        axes = axes.flatten()

        for i, feat_idx in enumerate(range(start_feat, end_feat)):
            # Extract and interpolate feature
            single_feature = lr_feats[:, :, :, feat_idx].numpy()
            feature_3d_full = interpolate_feature_to_full_size(
                single_feature, target_shape
            )
            max_proj = _get_z_range(feature_3d_full, z_range_name, target_shape)
            axes[i].imshow(max_proj, cmap="gray")
            axes[i].set_title(f"Feature {feat_idx}")
        for i in range(end_feat - start_feat, len(axes)):
            axes[i].axis("off")
        grid_dir_name = Path(f"feature_visualization/grids_{z_range_name}")
        grid_dir_name.mkdir(parents=True, exist_ok=True)
        grid_path = grid_dir_name.joinpath(f"grid_{page_idx}.png")
        fig.savefig(grid_path)
        plt.close(fig)


def find_interesting_features(
    lr_feats: torch.Tensor,
    top_k: int = 20,
    attention_heads: int = 6,
    save_path: Path = Path("feature_visualization/detailed_analysis"),
    save_pdf: bool = True,
    target_shape: Tuple[int, int, int] = (60, 608, 546),
    show_iteration_progress: bool = False,
) -> None:
    Z, Y, X, num_features = lr_feats[..., :-attention_heads].shape
    feature_variances = []

    for feat_idx in tqdm(range(num_features), desc="Finding interesting features"):
        single_feature = lr_feats[:, :, :, feat_idx].numpy()
        var = np.var(single_feature)
        feature_variances.append((feat_idx, var))

    # Sort by variance (descending)
    feature_variances.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {top_k} most variable features:")
    with PdfPages(save_path.joinpath("detailed_analysis.pdf")) as pdf:
        for i, (feat_idx, var) in tqdm(
            enumerate(feature_variances[:top_k]),
            desc="Analyzing features & Saving to PDF",
        ):
            if show_iteration_progress:
                print(f"{i + 1:2d}. Feature {feat_idx:3d}: variance = {var:.6f}")

            single_feature = lr_feats[:, :, :, feat_idx].numpy()
            feature_3d_full = interpolate_feature_to_full_size(
                single_feature, target_shape
            )
            io.imsave(
                save_path.joinpath(f"single_feature_{feat_idx}.tif"),
                feature_3d_full.astype(np.float16),
            )
            if save_pdf:
                _analyze_single_feature(
                    feature_3d_full=feature_3d_full,
                    variance=var,
                    feature_idx=feat_idx,
                    pdf=pdf,
                )
    if not save_pdf:
        for i, (feat_idx, var) in tqdm(
            enumerate(feature_variances[:top_k]),
            desc="Analyzing features",
        ):
            print(f"{i + 1:2d}. Feature {feat_idx:3d}: variance = {var:.6f}")


def _analyze_single_feature(
    feature_3d_full: np.ndarray,
    variance: float,
    feature_idx: int,
    pdf: PdfPages,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle(
        f"Detailed Analysis of Feature {feature_idx} - Variance: {variance:.6f}",
        fontsize=16,
        color="white",
    )

    axes[0].imshow(np.max(feature_3d_full, axis=0), cmap="viridis")
    axes[0].set_title("Max Projection (Z-axis)", color="white")
    axes[0].set_xlabel("X", color="white")
    axes[0].set_ylabel("Y", color="white")

    axes[1].imshow(np.max(feature_3d_full, axis=1), cmap="viridis")
    axes[1].set_title("Max Projection (Y-axis)", color="white")
    axes[1].set_xlabel("X", color="white")
    axes[1].set_ylabel("Z", color="white")

    axes[2].imshow(np.max(feature_3d_full, axis=2), cmap="viridis")
    axes[2].set_title("Max Projection (X-axis)", color="white")
    axes[2].set_xlabel("Y", color="white")
    axes[2].set_ylabel("Z", color="white")

    axes[3].hist(feature_3d_full.flatten(), bins=50, alpha=0.7, color="cyan")
    axes[3].set_title("Value Distribution", color="white")
    axes[3].set_xlabel("Feature Value", color="white")
    axes[3].set_ylabel("Frequency", color="white")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _generate_instance_seg_pdf(
    file_paths: List[Path],
    save_path: Path,
    grid_cols: int = 4,
    grid_rows: int = 3,
    figsize: Tuple[int, int] = (20, 15),
    save_name: str = "volume_instance_segmentation.pdf",
) -> None:
    items_per_page = grid_cols * grid_rows
    num_pages = (len(file_paths) + items_per_page - 1) // items_per_page
    save_path.mkdir(parents=True, exist_ok=True)
    pdf_path = save_path.joinpath(save_name)

    with PdfPages(pdf_path) as pdf:
        for page_idx in tqdm(range(num_pages), desc="Generating PDF pages"):
            start_idx = page_idx * items_per_page
            end_idx = min((page_idx + 1) * items_per_page, len(file_paths))
            page_paths = file_paths[start_idx:end_idx]

            fig, axes = plt.subplots(
                grid_rows, grid_cols * 2, figsize=(figsize[0], figsize[1])
            )

            if grid_rows == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(
                f"Volume and Instance Segmentation Comparison - Page {page_idx + 1}/{num_pages}",
                fontsize=16,
                color="white",
            )

            for i, file_path in enumerate(page_paths):
                row = i // grid_cols
                col = i % grid_cols

                try:
                    volume = io.imread(file_path.joinpath("volume_unnorm.tif"))
                    instance_seg = io.imread(file_path.joinpath("instance_seg.tif"))

                    # Volume subplot (left column)
                    vol_ax = axes[row, col * 2]
                    vol_ax.imshow(np.max(volume, axis=0), cmap="turbo")
                    vol_ax.set_title("Volume", color="white", fontsize=10)
                    vol_ax.set_xlabel("X", color="white", fontsize=8)
                    vol_ax.set_ylabel("Y", color="white", fontsize=8)
                    vol_ax.tick_params(colors="white", labelsize=8)

                    # Instance segmentation subplot (right column)
                    seg_ax = axes[row, col * 2 + 1]
                    seg_ax.imshow(np.max(instance_seg, axis=0))
                    seg_ax.set_title("Instance Seg", color="white", fontsize=10)
                    seg_ax.set_xlabel("X", color="white", fontsize=8)
                    seg_ax.set_ylabel("Y", color="white", fontsize=8)
                    seg_ax.tick_params(colors="white", labelsize=8)

                except Exception as e:
                    # Handle missing files or loading errors
                    vol_ax = axes[row, col * 2]
                    seg_ax = axes[row, col * 2 + 1]

                    vol_ax.text(
                        0.5,
                        0.5,
                        f"Error loading\n{file_path.name}\n{str(e)}",
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=10,
                    )
                    seg_ax.text(
                        0.5,
                        0.5,
                        f"Error loading\n{file_path.name}\n{str(e)}",
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=10,
                    )

                    vol_ax.set_title(
                        f"Volume - {file_path.name}", color="white", fontsize=10
                    )
                    seg_ax.set_title(
                        f"Instance Seg - {file_path.name}", color="white", fontsize=10
                    )

            for i in range(len(page_paths), items_per_page):
                row = i // grid_cols
                col = i % grid_cols
                axes[row, col * 2].axis("off")
                axes[row, col * 2 + 1].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved to: {pdf_path}")


def create_feature_pdf(
    lr_feats: torch.Tensor,
    target_shape: Tuple[int, int, int] = (60, 608, 546),
    output_pdf_path: Path = Path("features.pdf"),
    single_page: bool = False,
) -> Path:
    Z_lr, Y_lr, X_lr, num_features = lr_feats.shape

    with PdfPages(output_pdf_path) as pdf:
        if single_page:
            # Single page: 30 rows x 13 cols = 390 features exactly
            print(
                f"Creating single-page PDF with 30x13 grid for all {num_features} features"
            )

            fig, axes = plt.subplots(30, 13, figsize=(26, 60))
            fig.suptitle(
                f"All {num_features} Features - Max Projections",
                fontsize=24,
                color="white",
                y=0.995,
            )

            axes_flat = axes.flatten()

            for feat_idx in tqdm(range(num_features), desc="Processing features"):
                single_feature = lr_feats[:, :, :, feat_idx].numpy()
                feature_3d_full = interpolate_feature_to_full_size(
                    single_feature, target_shape
                )
                max_proj = np.max(feature_3d_full, axis=0)

                ax = axes_flat[feat_idx]
                ax.imshow(max_proj, cmap="viridis")
                ax.set_title(f"F{feat_idx}", fontsize=6, color="white")
                ax.set_xticks([])
                ax.set_yticks([])

                # Remove spines for cleaner look
                for spine in ax.spines.values():
                    spine.set_visible(False)

            plt.tight_layout()
            plt.subplots_adjust(top=0.985, hspace=0.2, wspace=0.05)
            pdf.savefig(fig, dpi=150, facecolor="black")
            plt.close(fig)

        else:
            # Multi-page: 20 features per page (4x5 grid)
            features_per_page = 20
            total_pages = (num_features + features_per_page - 1) // features_per_page

            print(
                f"Creating multi-page PDF: {total_pages} pages, {features_per_page} features per page"
            )

            for page_idx in range(total_pages):
                start_feat = page_idx * features_per_page
                end_feat = min((page_idx + 1) * features_per_page, num_features)

                print(
                    f"  Page {page_idx + 1}/{total_pages}: Features {start_feat}-{end_feat - 1}"
                )

                fig, axes = plt.subplots(4, 5, figsize=(15, 12))
                fig.suptitle(
                    f"Features {start_feat}-{end_feat - 1} (Page {page_idx + 1}/{total_pages})",
                    fontsize=16,
                    color="white",
                )

                axes_flat = axes.flatten()

                for i, feat_idx in enumerate(range(start_feat, end_feat)):
                    single_feature = lr_feats[:, :, :, feat_idx].numpy()
                    feature_3d_full = interpolate_feature_to_full_size(
                        single_feature, target_shape
                    )
                    max_proj = np.max(feature_3d_full, axis=0)

                    ax = axes_flat[i]
                    im = ax.imshow(max_proj, cmap="viridis")
                    ax.set_title(f"Feature {feat_idx}", fontsize=12, color="white")
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add colorbar for better interpretation
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Hide unused subplots
                for i in range(end_feat - start_feat, len(axes_flat)):
                    axes_flat[i].axis("off")

                plt.tight_layout()
                pdf.savefig(fig, dpi=150, facecolor="black")
                plt.close(fig)

    print(f"✓ PDF saved: {output_pdf_path}")
    return output_pdf_path


def find_closest_particles_to_points(
    track_to_study: pd.DataFrame,
    track_features: pd.DataFrame,
    t0: Tuple[float, float, float],
    t1: Tuple[float, float, float],
    max_distance: float = 15.0,
    suppress_prints: bool = False,
    feature_columns: List[str] = [],
    t_val_to_add: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], int | None]:
    results = {}
    closest_particle_at_t1 = None
    for timepoint, coords in [("t0", t0), ("t1", t1)]:
        t_val = 0 if timepoint == "t0" else 1
        t_val += t_val_to_add
        timepoint_particles = track_features[track_features["t"] == t_val]

        particle_coords = timepoint_particles[["x", "y", "z"]].values
        distances = cdist([coords], particle_coords, metric="euclidean")[0]

        valid_mask = distances <= max_distance
        if not np.any(valid_mask):
            if not suppress_prints:
                print(f"No particles within {max_distance} pixels of {timepoint}")
            results[timepoint] = {"particles": [], "distances": [], "features": []}
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_distances = distances[valid_mask]
        sorted_indices = np.argsort(valid_distances)

        sorted_particles = timepoint_particles.iloc[valid_indices[sorted_indices]]
        sorted_distances = valid_distances[sorted_indices]

        if not feature_columns:
            feature_columns = [
                col for col in timepoint_particles.columns if col.startswith("feature_")
            ]
        sorted_features = sorted_particles[feature_columns].values

        results[timepoint] = {
            "particles": sorted_particles,
            "distances": sorted_distances,
            "features": sorted_features,
        }
        if not suppress_prints:
            print(
                f"\n{timepoint.upper()} - Found {len(sorted_distances)} particles within {max_distance} pixels:"
            )
        if timepoint == "t1":
            closest_particle_at_t1 = sorted_particles.index[0]

        if not suppress_prints:
            # print(f"Closest particle at next timepoint: {closest_particle_at_t1}")
            for i, (idx, dist) in enumerate(
                zip(sorted_particles.index, sorted_distances)
            ):
                print(f"  {i + 1}. Particle {idx}: distance = {dist:.3f} pixels")

    return results, closest_particle_at_t1


def visualize_zoomed_region(
    file_paths: List[Path],
    time_point: int,
    closest_particle_in_t2: pd.DataFrame,
    plot_name: str,
) -> None:
    volume = io.imread(file_paths[time_point].joinpath("volume_unnorm.tif"))
    centroids = closest_particle_in_t2[["z", "y", "x"]].values
    if len(centroids) > 0:
        padding = 50
        x_coords = centroids[:, 2]
        y_coords = centroids[:, 1]
        z_coords = centroids[:, 0]
        x_min, x_max = (
            int(max(0, x_coords.min() - padding)),
            int(min(volume.shape[2], x_coords.max() + padding)),
        )
        y_min, y_max = (
            int(max(0, y_coords.min() - padding)),
            int(min(volume.shape[1], y_coords.max() + padding)),
        )
        z_min, z_max = (
            int(max(0, z_coords.min() - padding)),
            int(min(volume.shape[0], z_coords.max() + padding)),
        )

        print(
            f"Zoom region - X: [{x_min}:{x_max}], Y: [{y_min}:{y_max}], Z: [{z_min}:{z_max}]"
        )
        volume_zoomed = volume[z_min:z_max, y_min:y_max, x_min:x_max]

        centroids_zoomed = centroids.copy()
        centroids_zoomed[:, 0] -= z_min
        centroids_zoomed[:, 1] -= y_min
        centroids_zoomed[:, 2] -= x_min

        print(f"Zoomed volume shape: {volume_zoomed.shape}")
        fig = plt.figure(figsize=(24, 12))
        fig.patch.set_facecolor("black")
        ax1 = plt.subplot(1, 2, 1)
        stackview.imshow(
            volume,
            plot=ax1,
            colormap="turbo",
        )
        ax1.scatter(
            centroids[:, 2],
            centroids[:, 1],
            color="red",
            s=15,
            alpha=0.9,
            edgecolors="white",
            linewidths=1,
            marker="o",
            label=f"{len(centroids)} particles",
        )

        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
            alpha=0.8,
        )
        ax1.add_patch(rect)

        ax1.set_title(
            f"Full Volume Overview - {plot_name}",
            fontsize=14,
            color="white",
            fontweight="bold",
        )
        ax1.set_xlabel("X coordinate", fontsize=12, color="white")
        ax1.set_ylabel("Y coordinate", fontsize=12, color="white")
        ax1.tick_params(colors="white")
        ax2 = plt.subplot(1, 2, 2)
        stackview.imshow(
            volume_zoomed,
            plot=ax2,
            colormap="turbo",
        )

        ax2.scatter(
            centroids_zoomed[:, 2],
            centroids_zoomed[:, 1],
            color="red",
            s=40,
            alpha=0.9,
            edgecolors="white",
            linewidths=2,
            marker="o",
            zorder=10,
        )

        for i, (z, y, x) in enumerate(centroids_zoomed):
            z_original, y_original, x_original = centroids[i]
            label = f"{i + 1} - {z_original}, {y_original}, {x_original}"
            text = ax2.text(
                x - 1.5,
                y - 1.5,
                label,
                fontsize=10,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="none",
                    alpha=0.8,
                    edgecolor="red",
                    linewidth=1,
                ),
                zorder=11,
            )

        ax2.set_title(
            f"Zoomed Region - {len(centroids)} Particles - {plot_name}",
            fontsize=14,
            color="white",
            fontweight="bold",
        )
        ax2.set_xlabel("X coordinate (zoomed)", fontsize=12, color="white")
        ax2.set_ylabel("Y coordinate (zoomed)", fontsize=12, color="white")
        ax2.tick_params(colors="white")

        plt.tight_layout()
