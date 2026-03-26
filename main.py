import os
from src.process_faces import process_image
from src.clustering import cluster_embeddings, build_clusters, compute_centroids
from src.qdrant_db import create_collection, insert_vector
from config import FRAMES_PATH
from src.search import search_face
from src.process_faces import process_image

all_embeddings = []

def 
():
    global all_embeddings

    for file in os.listdir(FRAMES_PATH):
        path = os.path.join(FRAMES_PATH, file)

        embeddings = process_image(path)

        for emb in embeddings:
            all_embeddings.append(emb)

    print(f"Collected {len(all_embeddings)} embeddings")


def run_clustering():
    labels = cluster_embeddings(all_embeddings)
    for label in labels:
        print(label)

    clusters = build_clusters(all_embeddings, labels)

    centroids = compute_centroids(clusters)

    print(f"Found {len(centroids)} identities")

    return centroids

def store_centroids(centroids):
    create_collection()

    for label, centroid in centroids.items():
        insert_vector(
            vector=centroid,
            metadata={"person_id": int(label)},
            idx=int(label)
        )

    print("Stored all identities in Qdrant")
    
def recognize_from_frames():
    for file in os.listdir(FRAMES_PATH):
        path = os.path.join(FRAMES_PATH, file)

        embeddings = process_image(path)

        for emb in embeddings:
            results = search_face(emb)

            print("\nMatches:")
            for r in results:
                print(f"Person ID: {r.payload['person_id']}, Score: {r.score}")


if __name__ == "__main__":
    collect_embeddings()
    centroids = run_clustering()
    store_centroids(centroids)