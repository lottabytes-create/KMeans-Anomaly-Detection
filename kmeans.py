
import numpy as np


def kmeans(data, k, max_iters=100, tol=1e-4, random_state=42):
    """
    KMeans clustering implemented using NumPy.

    Parameters:
        data: Input dataset (2D NumPy array)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Convergence tolerance
        random_state: Random seed

    Returns:
        labels: Cluster assignment for each data point
        centroids: Final cluster centers
    """

    # Check input data
    data = np.asarray(data, dtype=float)

    if data.ndim != 2 or len(data) == 0:
        raise ValueError("Data must be a non-empty 2D array.")

    if not 1 <= k <= len(data):
        raise ValueError("k must be between 1 and the number of samples.")

    # Step 1: Initialize K centroids randomly
    rng = np.random.default_rng(random_state)

    initial_indices = rng.choice(
        len(data),
        size=k,
        replace=False
    )

    centroids = data[initial_indices].copy()

    # Step 2, 3 and 4: Iterate
    for iteration in range(max_iters):

        # Calculate distances between every point and centroid
        distances = np.linalg.norm(
            data[:, np.newaxis, :] - centroids[np.newaxis, :, :],
            axis=2
        )

        # Assign each point to its nearest centroid
        labels = np.argmin(distances, axis=1)

        # Store new centroids
        new_centroids = np.zeros_like(centroids)

        for cluster_index in range(k):

            # Select points belonging to this cluster
            cluster_points = data[labels == cluster_index]

            if len(cluster_points) > 0:

                # Calculate the average position
                new_centroids[cluster_index] = np.mean(
                    cluster_points,
                    axis=0
                )

            else:
                # Handle an empty cluster
                nearest_distances = np.min(distances, axis=1)

                farthest_index = np.argmax(nearest_distances)

                new_centroids[cluster_index] = data[farthest_index]

        # Step 5: Check convergence
        centroid_shifts = np.linalg.norm(
            new_centroids - centroids,
            axis=1
        )

        centroids = new_centroids

        if np.all(centroid_shifts < tol):
            print(f"KMeans converged at iteration {iteration + 1}")
            break

    # Recalculate final cluster assignments
    final_distances = np.linalg.norm(
        data[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )

    labels = np.argmin(final_distances, axis=1)

    return labels, centroids
