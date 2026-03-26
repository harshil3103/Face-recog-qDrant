import face_recognition
import cv2

def detect_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)

    faces = []
    for (top, right, bottom, left) in locations:
        face = image[top:bottom, left:right]
        faces.append(face)

    return faces