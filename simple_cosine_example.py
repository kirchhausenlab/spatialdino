#!/usr/bin/env python3
"""
Simple example showing how to use the new cosine similarity functions
with your existing code.
"""

# Just modify your existing code like this:

# BEFORE (your original code):
"""
features_wins_ids = plot_feature_comparisons(
    closest_particles=closest_particles,
    save_path=Path(f"t={gt_t}").joinpath(save_path),
    closest_particle_at_t1=closest_particle_at_t1,
    type_of_features="feature",
    save_pdf=False,
    feature_columns=feature_median_columns,
    comparison_method="cosine",
)
"""

# AFTER (enhanced with new method):
"""
# Method 1: Use the new ranking method for detailed analysis
features_wins_ids = plot_feature_comparisons(
    closest_particles=closest_particles,
    save_path=Path(f"t={gt_t}").joinpath(save_path),
    closest_particle_at_t1=closest_particle_at_t1,
    type_of_features="feature",
    save_pdf=False,
    feature_columns=feature_median_columns,
    comparison_method="cosine_ranking",  # 🆕 New method!
    min_similarity_threshold=0.75,      # 🆕 Configurable threshold
)

# Method 2: Get just the best match quickly
from src.cell_interactome.utils.tracking.tracking_utils import find_best_particle_match

best_match = find_best_particle_match(
    closest_particles=closest_particles,
    feature_columns=feature_median_columns,
    min_similarity_threshold=0.75,
    spatial_closest_id=closest_particle_at_t1,
    verbose=True
)

print(f"🎯 Recommended particle: {best_match['best_particle_id']}")
print(f"   Similarity: {best_match['best_similarity']:.6f}")
print(f"   Reliable: {best_match['is_reliable']}")
print(f"   Agrees with spatial: {best_match['spatial_agrees']}")
"""

print("Copy the AFTER code into your existing loop to enhance particle separation!")
print("\nKey benefits:")
print("✅ Clear ranking by feature similarity")
print("✅ Confidence scoring to identify ambiguous cases")
print("✅ Threshold-based filtering")
print("✅ Easy comparison with spatial distance results")
print("✅ Works with your existing data structures")
