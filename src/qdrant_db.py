from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import *

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        ),
    )
    print("Collection created")

def insert_vector(vector, metadata, idx):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {
                "id": idx,
                "vector": vector.tolist(),
                "payload": metadata
            }
        ]
    )