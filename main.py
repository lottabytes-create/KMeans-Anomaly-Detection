
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from kmeans import kmeans


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

# Automatically create the outputs directory
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 1: Load the dataset
# ==========================================

iris = load_iris()

# Use numerical features only.
# Species labels are deliberately ignored.
data = iris.data

feature_names = iris.feature_names

print("=" * 55)
print("KMEANS ANOMALY DETECTION")
print("=" * 55)

print("\nDataset: Iris")
print("Samples:", data.shape[0])
print("Features:", data.shape[1])


# ==========================================
# STEP 2: Standardize the data
# ==========================================

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# Prevent division by zero
std[std == 0] = 1

scaled_data = (data - mean) / std

print("\nData preprocessing completed.")


# ==========================================
# STEP 3: Perform KMeans clustering
# ==========================================

k = 3

labels, centroids = kmeans(
    scaled_data,
    k=k,
    max_iters=100,
    random_state=42
)

print("\nKMeans clustering completed.")
print("Number of clusters:", k)


# ==========================================
# STEP 4: Calculate distances
# ==========================================

assigned_centroids = centroids[labels]

distances = np.linalg.norm(
    scaled_data - assigned_centroids,
    axis=1
)


# ==========================================
# STEP 5: Cluster-specific anomaly detection
# ==========================================

thresholds = np.zeros(k)

anomaly_mask = np.zeros(len(data), dtype=bool)

for cluster_id in range(k):

    # Get distances belonging to this cluster
    cluster_distances = distances[labels == cluster_id]

    # Calculate median
    median_distance = np.median(cluster_distances)

    # Calculate Median Absolute Deviation (MAD)
    mad = np.median(
        np.abs(cluster_distances - median_distance)
    )

    # Establish a robust threshold
    if mad > 1e-12:
        threshold = median_distance + 3 * 1.4826 * mad
    else:
        # Fallback when MAD is approximately zero
        threshold = np.percentile(cluster_distances, 97.5)

    thresholds[cluster_id] = threshold

    # Identify anomalies in this cluster
    cluster_mask = labels == cluster_id

    anomaly_mask[cluster_mask] = (
        distances[cluster_mask] > threshold
    )


# Get anomaly indices
anomaly_indices = np.where(anomaly_mask)[0]

print("\n" + "=" * 55)
print("ANOMALY DETECTION RESULTS")
print("=" * 55)

print("Total samples:", len(data))
print("Detected anomalies:", len(anomaly_indices))

for cluster_id in range(k):

    cluster_size = np.sum(labels == cluster_id)

    anomaly_count = np.sum(
        anomaly_mask & (labels == cluster_id)
    )

    print(f"\nCluster {cluster_id}:")
    print("  Samples:", cluster_size)
    print("  Threshold:", round(thresholds[cluster_id], 4))
    print("  Anomalies:", anomaly_count)


# Display anomaly details
print("\nPotential anomalous samples:")

for index in anomaly_indices:

    print(
        f"Sample {index:3d} | "
        f"Cluster {labels[index]} | "
        f"Distance {distances[index]:.4f} | "
        f"Threshold {thresholds[labels[index]]:.4f}"
    )


# ==========================================
# STEP 6: Export detection results
# ==========================================

# Convert standardized centroids back to original scale
original_centroids = centroids * std + mean

# Save all samples, not only anomalies
results = np.column_stack((
    np.arange(len(data)),
    labels,
    distances,
    thresholds[labels],
    anomaly_mask.astype(int),
    data
))

header = (
    "sample_index,cluster,distance,threshold,is_anomaly,"
    + ",".join(feature_names)
)

csv_path = OUTPUT_DIR / "anomaly_results.csv"

np.savetxt(
    csv_path,
    results,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.6f"
)

print(f"\nResults exported to: {csv_path}")


# ==========================================
# STEP 7: Visualize the results
# ==========================================

plt.figure(figsize=(11, 7))

# Normal samples
plt.scatter(
    data[~anomaly_mask, 2],
    data[~anomaly_mask, 3],
    c=labels[~anomaly_mask],
    cmap="viridis",
    alpha=0.7,
    s=55,
    label="Normal Samples"
)

# Potential anomalies
plt.scatter(
    data[anomaly_mask, 2],
    data[anomaly_mask, 3],
    color="red",
    marker="x",
    s=110,
    linewidths=2,
    label="Potential Anomalies"
)

# Cluster centroids
plt.scatter(
    original_centroids[:, 2],
    original_centroids[:, 3],
    color="black",
    marker="*",
    s=250,
    label="Centroids"
)

plt.title(
    "KMeans Anomaly Detection - Iris Dataset",
    fontsize=14
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# Save visualization inside outputs/
image_path = OUTPUT_DIR / "anomaly_detection_result.png"

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Graph exported to: {image_path}")

plt.show()


# ==========================================
# EXPERIMENT COMPLETED
# ==========================================

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 55)

print(f"Output directory: {OUTPUT_DIR}")
print("Generated files:")
print("  1. anomaly_detection_result.png")
print("  2. anomaly_results.csv")
