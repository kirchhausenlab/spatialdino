from cell_interactome.utils.tracking.tracking_utils import postprocess_tracking
import pandas as pd
import matplotlib.pyplot as plt

# Process the tracking data
linked = postprocess_tracking(dino_long_tracks)

# Get track length for each unique track ID
track_lengths = linked["track_length"].groupby(linked["track_id"]).first()

# Get all unique track IDs
unique_track_ids = track_lengths.index.tolist()
print(f"Total number of unique tracks: {len(unique_track_ids)}")
print(f"Unique track IDs: {unique_track_ids}")

# Count tracks by length (value_counts automatically sorts by index)
length_counts = track_lengths.value_counts().sort_index(ascending=False)
print("\nTrack length distribution:")
print("Length -> Number of tracks")
print("-" * 30)
for length, count in length_counts.items():
    print(f"{length:6d} -> {count:4d} tracks")

# Also create a summary DataFrame for easier analysis
summary_df = pd.DataFrame({
    "track_length": length_counts.index,
    "number_of_tracks": length_counts.values,
}).sort_values("track_length", ascending=False)

print(f"\nSummary DataFrame:")
print(summary_df)

# Show some basic statistics
print(f"\nStatistics:")
print(f"Longest track: {track_lengths.max()} frames")
print(f"Shortest track: {track_lengths.min()} frames")
print(f"Average track length: {track_lengths.mean():.2f} frames")
print(f"Median track length: {track_lengths.median():.2f} frames")

# Plot the track length distribution
plt.figure(figsize=(12, 6))

# Create bar plot
plt.bar(length_counts.index, length_counts.values, edgecolor="black", alpha=0.7)

# Add value labels on top of bars
for length, count in length_counts.items():
    plt.text(length, count + 0.1, str(count), ha="center", va="bottom", fontsize=9)

plt.xlabel("Track Length (frames)")
plt.ylabel("Number of Tracks")
plt.title("Distribution of Track Lengths")
plt.grid(True, alpha=0.3)

# Reverse x-axis to show longest tracks first (if desired)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()

# Alternative: Line plot for smoother visualization
plt.figure(figsize=(12, 6))
plt.plot(
    length_counts.index, length_counts.values, marker="o", linewidth=2, markersize=6
)
plt.fill_between(length_counts.index, length_counts.values, alpha=0.3)

plt.xlabel("Track Length (frames)")
plt.ylabel("Number of Tracks")
plt.title("Track Length Distribution (Line Plot)")
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()
