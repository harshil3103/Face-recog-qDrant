from sklearn.cluster import DBSCAN
import numpy as np

def cluster_embeddings(embeddings):
    embeddings = np.array(embeddings)

    clustering = DBSCAN(
        eps=0.5,        # distance threshold for clustering(TUNE AS NEEDED)
        min_samples=5,  # minimum faces per person
        metric='cosine'
    ).fit(embeddings)

    return clustering.labels_


def build_clusters(embeddings, labels):
    clusters = {}

    for emb, label in zip(embeddings, labels):
        if label == -1:
            continue  # ignore noise

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(emb)

    return clusters

def compute_centroids(clusters):
    centroids = {}

    for label, vectors in clusters.items():
        centroids[label] = np.mean(vectors, axis=0)

    return centroids