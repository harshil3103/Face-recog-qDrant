import cv2
from src.embed_faces import get_embeddings

def process_image(image_path):
    img = cv2.imread(image_path)

    embeddings = get_embeddings(img)

    return embeddings