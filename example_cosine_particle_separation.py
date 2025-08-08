#!/usr/bin/env python3
"""
Example usage of improved cosine similarity for particle separation.

This demonstrates how to use the new functions to better separate particles
based on feature similarity rather than just spatial distance.
"""

from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np

# Assuming your existing imports and setup
from src.cell_interactome.utils.tracking.tracking_utils import (
    find_best_particle_match,
    rank_particles_by_cosine_similarity,
    plot_feature_comparisons,
)
from src.cell_interactome.visualize.utils import (
    find_closest_particles_to_points,
    filter_particles_at_t,
)


def example_improved_particle_matching():
    """
    Example of how to use the improved cosine similarity functions.
    Replace these with your actual data loading.
    """

    # Your existing setup (replace with actual data)
    # track_features = pd.read_csv("your_track_features.csv")
    # file_paths = ["path/to/your/files"]
    # feature_median_columns = ["feature_0", "feature_1", ...]  # Your feature columns

    print("🔬 IMPROVED PARTICLE SEPARATION WITH COSINE SIMILARITY")
    print("=" * 60)

    # Example parameters (adjust these for your data)
    gt_range = [i for i in range(0, 24)]
    t_range = [i for i in range(1, 25)]
    max_distance = 16.0
    closest_coordinates = [38, 394, 304]  # t=4, x,y,z

    # Your feature columns (replace with actual)
    feature_median_columns = [f"feature_{i}" for i in range(390)]

    # Set up filter function (replace with your actual function)
    # filter_particle_fn = partial(
    #     filter_particles_at_t, track_features=track_features, range_value=15
    # )

    # Example particle IDs to test
    ids_lst = [3, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 3, 2, 2, 3, 4, 4, 4, 4, 5, 5, 3, 6]

    results_log = []

    for i, (gt_t, t_next, closest_id) in enumerate(
        zip(gt_range[:5], t_range[:5], ids_lst[:5])
    ):
        print(f"\n📍 TIMEPOINT {i + 1}: t={gt_t} -> t={t_next}")
        print("-" * 40)

        # Replace this section with your actual particle filtering and setup
        """
        z_gt, y_gt, x_gt = closest_coordinates
        particles_in_gt_roi = filter_particle_fn(t=gt_t, x_val=x_gt, y_val=y_gt)
        particles_in_roi = filter_particle_fn(t=t_next, x_val=x_gt, y_val=y_gt)
        particles_in_roi["particle_id"] = range(len(particles_in_roi))
        particles_in_roi["particle_id"] += 1
        
        t1 = (
            particles_in_roi[particles_in_roi["particle_id"] == closest_id][["x", "y", "z"]]
            .values[0]
            .tolist()
        )
        closest_coordinates = [t1[2], t1[1], t1[0]]
        
        closest_particles, closest_particle_at_t1 = find_closest_particles_to_points(
            track_to_study=particles_in_roi,
            track_features=track_features,
            t0=(x_gt, y_gt, z_gt),
            t1=t1,
            max_distance=max_distance,
            suppress_prints=False,
            t_val_to_add=gt_t,
            feature_columns=feature_median_columns,
        )
        """

        # For demonstration, create mock data structure
        # Replace this with your actual closest_particles data
        closest_particles = create_mock_closest_particles()  # You'll replace this
        closest_particle_at_t1 = 1944  # Replace with actual

        # 🎯 METHOD 1: Simple best match finder
        print("\n🎯 METHOD 1: Simple Best Match")
        best_match = find_best_particle_match(
            closest_particles=closest_particles,
            feature_columns=feature_median_columns,
            min_similarity_threshold=0.75,  # Adjust threshold as needed
            spatial_closest_id=closest_particle_at_t1,
            verbose=True,
        )

        # 🏆 METHOD 2: Detailed ranking
        print("\n🏆 METHOD 2: Detailed Ranking")
        ranking_results = rank_particles_by_cosine_similarity(
            closest_particles=closest_particles,
            feature_columns=feature_median_columns,
            min_similarity_threshold=0.70,
        )

        if "error" not in ranking_results:
            print(f"Top 3 candidates:")
            for j, particle in enumerate(ranking_results["rankings"][:3]):
                print(
                    f"  {j + 1}. Particle {particle['particle_id']}: "
                    f"similarity={particle['cosine_similarity']:.6f}, "
                    f"confidence={particle['confidence']:.3f}"
                )

        # 📊 METHOD 3: Enhanced visualization (your existing function with new method)
        print("\n📊 METHOD 3: Enhanced Visualization")
        features_wins_ids = plot_feature_comparisons(
            closest_particles=closest_particles,
            save_path=Path(f"t={gt_t}").joinpath("analysis_results"),
            closest_particle_at_t1=closest_particle_at_t1,
            type_of_features="feature",
            save_pdf=False,
            feature_columns=feature_median_columns,
            comparison_method="cosine_ranking",  # New method!
            min_similarity_threshold=0.75,
        )

        # Log results for analysis
        result = {
            "timepoint": f"t{gt_t}->t{t_next}",
            "spatial_closest": closest_particle_at_t1,
            "feature_best": best_match.get("best_particle_id"),
            "feature_similarity": best_match.get("best_similarity"),
            "is_reliable": best_match.get("is_reliable"),
            "spatial_agrees": best_match.get("spatial_agrees"),
        }
        results_log.append(result)

    # 📈 SUMMARY ANALYSIS
    print(f"\n📈 SUMMARY ANALYSIS")
    print("=" * 60)

    total_cases = len(results_log)
    agreement_cases = sum(1 for r in results_log if r["spatial_agrees"])
    reliable_cases = sum(1 for r in results_log if r["is_reliable"])

    print(f"Total cases analyzed: {total_cases}")
    print(
        f"Spatial-Feature agreement: {agreement_cases}/{total_cases} ({agreement_cases / total_cases * 100:.1f}%)"
    )
    print(
        f"Reliable feature matches: {reliable_cases}/{total_cases} ({reliable_cases / total_cases * 100:.1f}%)"
    )

    # Cases where feature-based differs from spatial
    disagreement_cases = [r for r in results_log if not r["spatial_agrees"]]
    if disagreement_cases:
        print(f"\n⚠️  Cases where feature and spatial differ:")
        for case in disagreement_cases:
            print(
                f"  {case['timepoint']}: spatial={case['spatial_closest']}, "
                f"feature={case['feature_best']} (sim={case['feature_similarity']:.4f})"
            )


def create_mock_closest_particles():
    """Create mock data for demonstration. Replace with your actual data."""
    # This is just for the example - you'll use your real closest_particles data
    mock_data = {
        "t0": {
            "particles": pd.DataFrame(),  # Empty, will use features array
            "features": [np.random.rand(390)],  # Mock reference features
        },
        "t1": {
            "particles": pd.DataFrame({
                "particle_id": [1944, 1892, 1946, 2545],
                **{f"feature_{i}": np.random.rand(4) for i in range(390)},
            }).set_index("particle_id"),
            "features": np.random.rand(4, 390),  # Mock candidate features
        },
    }
    return mock_data


def adjust_similarity_threshold():
    """
    Helper function to find optimal similarity threshold for your data.
    """
    print("\n🔧 THRESHOLD OPTIMIZATION")
    print("=" * 40)

    # Test different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        # You would run your analysis with this threshold
        # and count reliable matches vs total matches

        # Mock results for demonstration
        reliable_matches = max(0, 10 - int((threshold - 0.5) * 20))
        total_matches = 10

        print(
            f"  Reliable matches: {reliable_matches}/{total_matches} "
            f"({reliable_matches / total_matches * 100:.1f}%)"
        )


if __name__ == "__main__":
    # Run the example (you'll need to adapt this to your actual data)
    print("This is an example script. Adapt it to your actual data setup.")
    print("Key improvements for particle separation:")
    print("1. 🎯 find_best_particle_match() - Simple interface")
    print("2. 🏆 rank_particles_by_cosine_similarity() - Detailed analysis")
    print("3. 📊 plot_feature_comparisons(..., method='cosine_ranking') - Enhanced viz")
    print("4. ⚙️  Configurable similarity thresholds")
    print("5. 🔍 Confidence scoring to identify ambiguous cases")

    # Uncomment when you have your data ready:
    # example_improved_particle_matching()
    # adjust_similarity_threshold()
