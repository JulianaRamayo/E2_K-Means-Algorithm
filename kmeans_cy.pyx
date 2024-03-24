# cython: profile=True

import numpy as np
cimport numpy as cnp
import random

cdef class Cy_KMeans:
    # Declaration of class attributes with C-level types for performance optimization.
    cdef int k, max_iterations
    cdef public centroids

    def __init__(self, int k, int max_iterations=100):
        """
        Initializes the KMeans object with the number of clusters 'k' and the maximum number of iterations.
        Also initializes the 'centroids' attribute as an empty NumPy array.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = np.empty((0, 0), dtype=np.float64)

    def fit(self, data):
        """
        Fits the KMeans model to the data, adjusting centroids to minimize within-cluster sum of squares.
        """
        cdef int n_samples = data.shape[0]
        # Ensure data is a NumPy array of the correct type at the start
        cdef cnp.ndarray data_np = np.asanyarray(data, dtype=np.float64)
        # Randomly initialize centroids.
        self.centroids = data_np[random.sample(range(n_samples), self.k), :]

        for _ in range(self.max_iterations):
            # Compute distances and assign points
            distances = np.zeros((n_samples, self.k), dtype=np.float64)
            for i in range(self.k):
                centroid = self.centroids[i]
                distances[:, i] = np.sum((data_np - centroid) ** 2, axis=1)
            
            closest_centroids = np.argmin(distances, axis=1) # Assign each point to the closest centroid.

            new_centroids = np.array([data_np[closest_centroids == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, data):
        """
        Predicts the closest cluster each sample in 'data' belongs to, based on the trained centroids.
        """
        cdef cnp.ndarray data_np = np.asanyarray(data, dtype=np.float64)
        # Compute distances from each point to each centroid.
        distances = np.zeros((data_np.shape[0], self.k), dtype=np.float64)
        for i in range(self.k):
            centroid = self.centroids[i]
            distances[:, i] = np.sum((data_np - centroid) ** 2, axis=1)
        # Return the index of the closest centroid for each sample.
        return np.argmin(distances, axis=1)

    def print_clusters(self, data):
        """
        Prints the clusters with their centroids and the points belonging to each cluster.
        """
        cdef cnp.ndarray data_np = np.asanyarray(data, dtype=np.float64)
        # Get cluster assignments for each point in 'data'.
        cluster_assignments = self.predict(data_np)
        for i in range(self.k):
            cluster_points = data_np[cluster_assignments == i]
            print(f"Cluster {i + 1} (Centroid: {self.centroids[i]}): {len(cluster_points)} points")
            for point in cluster_points:
                print(f"  {point}")
            print()  # Newline for readability