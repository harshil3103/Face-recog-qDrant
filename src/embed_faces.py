import insightface
import numpy as np

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0) 

def get_embeddings(image):
    faces = app.get(image)

    embeddings = []
    for face in faces:
        embeddings.append(face.embedding)

    return embeddings