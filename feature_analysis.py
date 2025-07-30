import numpy as np
import matplotlib.pyplot as plt

# Get the data
closest_particles_lst = closest_particles["t1"]
particles_df = closest_particles_lst["particles"]
features_lst = [
    "feature_{}".format(i) for i in range(384, 390)
]  # Last 6 features (384-389)

# Get particle 340 features from t0 (closest to t0)
particle_340_features = closest_particles["t0"]["features"][0]

# Calculate differences in a for loop and print them
print("=" * 80)
print("FEATURE DIFFERENCES: Particle 340 (t=0) vs Particles at t=1")
print("=" * 80)

# Store differences for plotting
all_differences = []

for i, particle_idx in enumerate(particles_df.index):
    particle_features = particles_df[features_lst].iloc[i].values
    differences = (
        particle_340_features[384:390] - particle_features
    )  # Only last 6 features

    print(
        f"\nParticle {particle_idx} (Distance: {closest_particles_lst['distances'][i]:.3f} pixels):"
    )
    print(f"  Mean difference: {np.mean(differences):.6f}")
    print(f"  Std difference: {np.std(differences):.6f}")
    print(f"  Max difference: {np.max(differences):.6f}")
    print(f"  Min difference: {np.min(differences):.6f}")

    # Find features with largest differences (all 6 features)
    largest_diff_indices = np.argsort(np.abs(differences))[-6:]
    print(f"  All 6 features differences:")
    for idx in largest_diff_indices:
        feature_num = 384 + idx
        print(f"    Feature {feature_num}: {differences[idx]:.6f}")

    all_differences.append(differences)

# Create a much larger plot with better visibility
fig, axes = plt.subplots(len(particles_df), 1, figsize=(20, 6 * len(particles_df)))
if len(particles_df) == 1:
    axes = [axes]

feature_indices = np.arange(384, 390)  # Last 6 features

for i, (particle_idx, differences) in enumerate(
    zip(particles_df.index, all_differences)
):
    ax = axes[i]

    # Plot with thicker lines and better colors
    ax.plot(
        feature_indices,
        differences,
        linewidth=2,
        alpha=0.8,
        label=f"Particle {particle_idx} (dist: {closest_particles_lst['distances'][i]:.3f})",
    )

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1)

    # Set better title and labels
    ax.set_title(
        f"Feature Differences: Particle 340 (t=0) vs Particle {particle_idx} (t=1) - Last 6 Features",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Feature Index", fontsize=12)
    ax.set_ylabel("Feature Difference (340 - {})".format(particle_idx), fontsize=12)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Set better y-axis limits to show the differences clearly
    y_range = np.max(np.abs(differences)) * 1.1
    ax.set_ylim(-y_range, y_range)

    # Add some statistics as text
    stats_text = f"Mean: {np.mean(differences):.4f}\nStd: {np.std(differences):.4f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

plt.tight_layout()
plt.show()

# Also create a summary plot showing all differences together
plt.figure(figsize=(20, 10))
for i, (particle_idx, differences) in enumerate(
    zip(particles_df.index, all_differences)
):
    plt.plot(
        feature_indices,
        differences,
        linewidth=2,
        alpha=0.8,
        label=f"Particle {particle_idx} (dist: {closest_particles_lst['distances'][i]:.3f})",
    )

plt.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1)
plt.title(
    "All Feature Differences: Particle 340 (t=0) vs All Particles at t=1 - Last 6 Features",
    fontsize=16,
    fontweight="bold",
)
plt.xlabel("Feature Index", fontsize=12)
plt.ylabel("Feature Difference", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
