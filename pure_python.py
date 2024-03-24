import random

class PP_KMeans:
    def __init__(self, k, max_iterations=100):
        # Initialize the KMeans clustering with specified number of clusters (k)
        # and the maximum number of iterations.
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []  # List to store the centroids of the clusters.
        self.clusters = [[] for _ in range(k)]  # List of lists to store the data points of each cluster.

    def fit(self, data):
        # Fit the model to the data.
        # Randomly select initial centroids from the data points.
        self.centroids = random.sample(data, self.k)
        
        # Iteratively refine the centroids.
        for _ in range(self.max_iterations):
            # Assign points to the nearest cluster.
            self.clusters = self._assign_points_to_clusters(data)
            
            # Update centroids based on current cluster assignments.
            new_centroids = self._update_centroids()
            
            # Check if centroids have changed - if not, the algorithm has converged.
            if self._check_convergence(new_centroids):
                break  # Exit the loop if converged.
            
            self.centroids = new_centroids  # Update centroids for next iteration.

    def _euclidean_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points.
        return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

    def _assign_points_to_clusters(self, data):
        # Assign each data point to the cluster with the nearest centroid.
        clusters = [[] for _ in self.centroids]
        for point in data:
            closest_centroid = min(range(len(self.centroids)),
                                    key=lambda i: self._euclidean_distance(point, self.centroids[i]))
            clusters[closest_centroid].append(point)
        return clusters

    def _update_centroids(self):
        # Update the centroids of each cluster based on the current assignments.
        return [
            [sum(dim) / len(cluster) for dim in zip(*cluster)]  # Calculate the mean for each dimension.
            if cluster else random.choice(self.centroids)  # Handle the case of an empty cluster.
            for cluster in self.clusters
        ]

    def _check_convergence(self, new_centroids):
        # Check if the centroids have changed since the last iteration.
        # If not, the algorithm has converged.
        return new_centroids == self.centroids

    def print_clusters(self):
        # Print the clusters with their centroids and points.
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i + 1} (Centroid: {self.centroids[i]}): {len(cluster)} points")
            for point in cluster:
                print(f"  {point}")
            print()  # Print a newline for readability between clusters.