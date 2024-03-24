import numpy as np
import numexpr as ne
import random

class NN_KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, data):
        data = np.array(data)
        n_samples = data.shape[0]

        # Randomly select initial centroids from the data points
        indices = random.sample(range(n_samples), self.k)
        self.centroids = data[indices, :]

        for _ in range(self.max_iterations):
            # Compute distances from each point to each centroid
            distances = np.zeros((n_samples, self.k))
            for i, centroid in enumerate(self.centroids):
                # Using numexpr to calculate the squared Euclidean distance
                distances[:, i] = ne.evaluate('sum((data - centroid) ** 2, axis=1)')
            
            # Assign each point to the closest centroid
            closest_centroids = np.argmin(distances, axis=1)

            # Update centroids to be the mean of points assigned to them
            new_centroids = np.array([data[closest_centroids == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, data):
        data = np.array(data)
        distances = np.zeros((data.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = ne.evaluate('sum((data - centroid) ** 2, axis=1)')
        return np.argmin(distances, axis=1)

    def print_clusters(self, data):
        data = np.array(data)
        cluster_assignments = self.predict(data)
        for i in range(self.k):
            cluster_points = data[cluster_assignments == i]
            print(f"Cluster {i + 1} (Centroid: {self.centroids[i]}): {len(cluster_points)} points")
            for point in cluster_points:
                print(f"  {point}")
            print()  # Newline for readability