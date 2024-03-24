import numpy as np
from memory_profiler import profile
from kmeans_cy import Cy_KMeans

# Generate data points
cy_data = np.random.randint(0, 101, size=(30000, 2))

# Create a cluster instance
cy_kmeans = Cy_KMeans(k=5)

@profile
def run_kmeans():
    # Execute the method you want to profile
    cy_kmeans.fit(cy_data)

if __name__ == "__main__":
    run_kmeans()