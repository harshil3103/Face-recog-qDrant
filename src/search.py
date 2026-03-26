from qdrant_client import QdrantClient
from config import *
from config import THRESHOLD
client = QdrantClient (host=QDRANT_HOST, port=QDRANT_PORT)

def search_face(vector):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector.tolist(),
        limit=3,
        search_params={"ef": 128}
    )

    for r in results:
        if r.score > THRESHOLD:
            print("Known:", r.payload)
        else:
            print("Unknown person")

    return results