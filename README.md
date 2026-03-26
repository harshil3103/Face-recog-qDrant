# Face Recognition Pipeline using Qdrant (InsightFace + DBSCAN)

## Overview
This project implements a face recognition and similarity search pipeline using vector embeddings and a vector database. It processes video input, extracts faces, generates embeddings using InsightFace, clusters identities using DBSCAN, and stores them in Qdrant for fast similarity-based retrieval.
The system is designed to simulate a real-world scenario such as surveillance or identity grouping, where labeled data is not available beforehand.


## Key Features
- Face detection and embedding using InsightFace (ArcFace-based embeddings)
- Frame extraction from video input using OpenCV
- Unsupervised identity clustering using DBSCAN
- Vector storage and similarity search using Qdrant
- Unknown face detection using similarity thresholding
- Stage-wise performance tracking (timing of each pipeline step)
- Modular and extensible pipeline design

---
## Architecture
Video Input
↓
Frame Extraction (OpenCV)
↓
Face Detection + Embedding (InsightFace)
↓
Clustering (DBSCAN)
↓
Centroid Generation
↓
Storage (Qdrant Vector DB)
↓
Similarity Search / Recognition

## Tech Stack
- Python
- OpenCV
- InsightFace (ArcFace embeddings)
- Scikit-learn (DBSCAN clustering)
- Qdrant (Vector Database)
- NumPy

### 2. Setup Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
---
### 3. Configure Paths
Update `config.py`:
- VIDEO_PATH → path to input video
- FRAMES_PATH → output frames directory
---
### 4. Run Pipeline
python main.py

---
### 5. Test with Single Image
python src/test_single_image.py

---
## Performance Insights
The pipeline includes stage-wise timing to identify bottlenecks:
- Frame extraction → fast
- Embedding generation → CPU bottleneck (InsightFace)
- Clustering → moderate (depends on data size)
- Qdrant operations → fast

---
## Limitations

- CPU-only inference leads to slower embedding generation
- DBSCAN does not scale well for very large datasets
- No ground truth labels (unsupervised clustering)
- Accuracy depends on video quality (lighting, angles, blur)
---

## Future Improvements
- Replace DBSCAN with HDBSCAN for better clustering
- Add GPU acceleration for faster embeddings
- Implement real-time processing pipeline
- Add visualization layer (matched faces display)
- Improve unknown face detection with adaptive thresholds
- Build a web interface for interaction

---

## Project Structure
Face-recog-qDrant
├── src/
│ ├── extract_frames.py
│ ├── process_faces.py
│ ├── embed_faces.py
│ ├── clustering.py
│ ├── qdrant_db.py
│ ├── search.py
│
├── main.py
├── config.py
├── requirements.txt
├── .gitignore
---
## Notes
- Dataset and video files are not included in this repository
- Users must provide their own video input
- InsightFace models are downloaded automatically during execution

---
## Summary
This project demonstrates how vector databases and deep learning embeddings can be combined to build a scalable face recognition system without labeled data. It highlights practical challenges such as performance bottlenecks, clustering accuracy, and system design trade-offs.
