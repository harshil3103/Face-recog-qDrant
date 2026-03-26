import cv2
import os
from config import VIDEO_PATH, FRAMES_PATH, FRAME_SKIP

def extract_frames():
    os.makedirs(FRAMES_PATH, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_SKIP == 0:
            file_path = os.path.join(FRAMES_PATH, f"frame_{saved}.jpg")
            cv2.imwrite(file_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Saved {saved} frames")

if __name__ == "__main__":
    extract_frames()