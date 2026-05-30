# K-means Clustering from Scratch
# Haochen Li - SMU Computer Engineering

import random
import math

def euclidean_distance(p1, p2):
    """Calculate distance between two points"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid"""
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        nearest = distances.index(min(distances))
        clusters[nearest].append(point)
    return clusters

def update_centroids(clusters):
    """Calculate new centroid as mean of each cluster"""
    new_centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            new_centroids.append(centroid)
    return new_centroids

def kmeans(data, k, max_iterations=100):
    """Run K-means clustering"""
    # Step 1: Randomly initialize centroids
    centroids = random.sample(data, k)

    for i in range(max_iterations):
        # Step 2: Assign points to clusters
        clusters = assign_clusters(data, centroids)

        # Step 3: Update centroids
        new_centroids = update_centroids(clusters)

        # Step 4: Check if converged
        if new_centroids == centroids:
            print(f"Converged after {i+1} iterations")
            break
        centroids = new_centroids

    return clusters, centroids

# Test data: 2D points (imagine x=hours studied, y=assignment score)
data = [
    [1, 50], [1.5, 55], [2, 52],   # Low group
    [5, 75], [5.5, 78], [6, 80],   # Mid group
    [9, 92], [9.5, 95], [10, 98]   # High group
]

# Run K-means with k=3
clusters, centroids = kmeans(data, k=3)

print("\nK-means Clustering Results:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")

print("\nFinal Centroids:")
for i, c in enumerate(centroids):
    print(f"Centroid {i+1}: [{c[0]:.2f}, {c[1]:.2f}]")