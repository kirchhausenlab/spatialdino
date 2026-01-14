#!/usr/bin/env python3
"""
Simple decay correction functions for integration into existing plotting code
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")


def exponential_decay(t, a, k):
    """Exponential decay function: a * exp(-k*t)"""
    return a * np.exp(-k * t)


def get_decay_correction_for_dino_track(dino_df, dino_id):
    """
    Get decay parameters for a specific DINO track

    Returns:
    dict with 'k', 'a', 'fit_success', 'r_squared'
    """
    try:
        track_data = dino_df[dino_df["ID"] == dino_id].sort_values("t")

        if len(track_data) < 3:
            return {"k": np.nan, "a": np.nan, "fit_success": False, "r_squared": np.nan}

        # Normalize time to start from 0
        time_points = track_data["t"].values - track_data["t"].min()
        intensities = track_data["intensity"].values

        # Fit exponential decay
        p0 = [max(intensities), 0.01]  # Initial guess
        popt, pcov = curve_fit(
            exponential_decay, time_points, intensities, p0=p0, maxfev=1000
        )
        a, k = popt

        # Calculate R²
        y_pred = exponential_decay(time_points, a, k)
        ss_res = np.sum((intensities - y_pred) ** 2)
        ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {"k": k, "a": a, "fit_success": True, "r_squared": r_squared}

    except:
        return {"k": np.nan, "a": np.nan, "fit_success": False, "r_squared": np.nan}


def apply_decay_correction_to_intensities(time_points, intensities, k_value):
    """
    Apply decay correction to intensity values

    Parameters:
    time_points: array of time points
    intensities: array of intensity values
    k_value: decay constant from exponential fit

    Returns:
    corrected_intensities: array of corrected intensity values
    """
    if np.isnan(k_value) or k_value <= 0:
        return intensities  # Return original if k is invalid

    # Normalize time to start from 0
    t_normalized = time_points - np.min(time_points)

    # Apply correction: divide by decay factor
    decay_factors = np.exp(-k_value * t_normalized)
    corrected_intensities = intensities / decay_factors

    return corrected_intensities


# Example of how to integrate into your existing plotting function:
def plot_cme_dino_analysis_with_decay_correction_simple(
    cme_id: int,
    dino_id: int,
    csv_file: str = "cme_dino_comprehensive_analysis.csv",
    name_plot_1: str = "CME ID",
    name_plot_2: str = "DINO ID",
    ratio_scale: tuple = (0.0, 0.3),
) -> None:
    """
    Modified version of your plotting function with decay correction
    """
    import matplotlib.pyplot as plt

    # Load data (as in your original function)
    cme_df = pd.read_csv("scripts/tracking/zeiss_560.csv")
    dino_df = pd.read_csv("scripts/tracking/zeiss_642.csv")

    comprehensive_data = pd.read_csv(csv_file)
    comparison_data = comprehensive_data[
        (comprehensive_data["cme_id"] == cme_id)
        & (comprehensive_data["best_dino_id"] == dino_id)
    ].copy()
    comparison_data = comparison_data.sort_values("t")

    # Get decay correction for this DINO track
    decay_params = get_decay_correction_for_dino_track(dino_df, dino_id)
    k_value = decay_params["k"]
    fit_success = decay_params["fit_success"]

    # Apply decay correction to DINO intensities
    if fit_success and not np.isnan(k_value) and k_value > 0:
        corrected_dino_intensities = apply_decay_correction_to_intensities(
            comparison_data["t"].values,
            comparison_data["dino_intensity"].values,
            k_value,
        )
        comparison_data["dino_intensity_corrected"] = corrected_dino_intensities
        print(
            f"Applied decay correction with k={k_value:.4f}, R²={decay_params['r_squared']:.3f}"
        )
    else:
        comparison_data["dino_intensity_corrected"] = comparison_data["dino_intensity"]
        print("No decay correction applied (fit failed or invalid k)")

    # Calculate corrected intensity ratio (smaller/larger)
    comparison_data["corrected_intensity_ratio"] = np.minimum(
        comparison_data["cme_intensity"], comparison_data["dino_intensity_corrected"]
    ) / np.maximum(
        comparison_data["cme_intensity"], comparison_data["dino_intensity_corrected"]
    )

    # Your existing plotting code with modifications...
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Plot 1: Combined intensity comparison - now shows original + corrected DINO
    ax1 = axes[0, 0]
    ax1.plot(
        comparison_data["t"],
        comparison_data["cme_intensity"],
        "r-",
        label=f"{name_plot_1}={cme_id}",
        marker="o",
    )
    ax1.plot(
        comparison_data["t"],
        comparison_data["dino_intensity"],
        "b-",
        label=f"{name_plot_2}={dino_id} (original)",
        marker="s",
        alpha=0.7,
    )
    ax1.plot(
        comparison_data["t"],
        comparison_data["dino_intensity_corrected"],
        "c-",
        label=f"{name_plot_2}={dino_id} (corrected)",
        marker="^",
        linewidth=2,
    )
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Intensity")
    title = "Combined Intensity Comparison"
    if fit_success:
        title += f"\nDINO decay: k={k_value:.4f}, R²={decay_params['r_squared']:.3f}"
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # Rest of your plots remain the same...
    # Plot 2: Distance over time (unchanged)
    ax2 = axes[0, 1]
    ax2.plot(comparison_data["t"], comparison_data["distance"], "g-", marker="d")
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Distance (units)")
    ax2.set_title("Distance Over Time")
    ax2.grid(True)

    # Plot 3: Now uses CORRECTED intensity ratio
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
    ax3.set_title("Decay-Corrected Intensity Ratio Over Time")
    ax3.set_ylim(ratio_scale)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plots 4-6: Add your existing trajectory and individual intensity plots...
    # (keeping them the same as your original function)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\nDecay Correction Summary:")
    print(f"DINO Track {dino_id}: k={k_value:.4f}, fit_success={fit_success}")
    print(
        f"Mean original DINO intensity: {comparison_data['dino_intensity'].mean():.1f}"
    )
    print(
        f"Mean corrected DINO intensity: {comparison_data['dino_intensity_corrected'].mean():.1f}"
    )
    print(
        f"Corrected intensity ratio (CME/DINO): {comparison_data['corrected_intensity_ratio'].mean():.3f}"
    )


if __name__ == "__main__":
    # Example usage
    print("Testing simple decay correction functions...")

    # Load DINO data
    dino_df = pd.read_csv("scripts/tracking/zeiss_642.csv")

    # Test decay correction for DINO track 0
    dino_id = 0
    decay_params = get_decay_correction_for_dino_track(dino_df, dino_id)
    print(f"\nDINO Track {dino_id} decay parameters:")
    print(f"  k = {decay_params['k']:.4f}")
    print(f"  a = {decay_params['a']:.1f}")
    print(f"  R² = {decay_params['r_squared']:.3f}")
    print(f"  Fit successful: {decay_params['fit_success']}")

    # Test correction on sample data
    track_data = dino_df[dino_df["ID"] == dino_id].sort_values("t")
    original_intensities = track_data["intensity"].values
    corrected_intensities = apply_decay_correction_to_intensities(
        track_data["t"].values, original_intensities, decay_params["k"]
    )

    print(f"\nSample correction results:")
    print(f"  Original intensity at t=0: {original_intensities[0]:.1f}")
    print(f"  Corrected intensity at t=0: {corrected_intensities[0]:.1f}")
    print(f"  Original intensity at t=20: {original_intensities[20]:.1f}")
    print(f"  Corrected intensity at t=20: {corrected_intensities[20]:.1f}")
    print(f"  Original intensity at t=39: {original_intensities[39]:.1f}")
    print(f"  Corrected intensity at t=39: {corrected_intensities[39]:.1f}")
