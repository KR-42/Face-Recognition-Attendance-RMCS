import os
import cv2
import numpy as np
from tkinter import messagebox
from utils import dataset_dir, face_cascade, embedder, detector

def register_face(name):
    if os.path.exists(os.path.join(dataset_dir, name)):
        messagebox.showerror("Error", "Nama sudah terdaftar. Gunakan nama lain.")
        return
    cam = cv2.VideoCapture(0)
    count = 0
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{person_dir}/{name}_{count}.jpg", face_img)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f"{count}/40", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow('Register - Press q to quit', frame)
        if count >= 40 or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"Captured {count} faces for {name}")
    def save_embeddings_for_user(name):
        person_dir = os.path.join(dataset_dir, name)
        embeddings = []
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path)
            emb = get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        np.save(os.path.join(person_dir, f'{name}_embedding.npy'), embeddings)
    save_embeddings_for_user(name)
def get_face_embedding(image):
    results = detect_faces_mtcnn(image)
    if results:
        x, y, w, h = results[0]['box']
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        emb = embedder.embeddings([face_rgb])[0]
        return emb
    return None
def get_batch_face_embeddings(faces):
    embeddings = []
    faces_rgb = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
    embeddings = embedder.embeddings(faces_rgb)  
    return embeddings
def detect_faces_mtcnn(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    denoised_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    results = detector.detect_faces(image)
    detected_faces = []
    for result in results:
        if result['confidence'] > 0.9:  
            detected_faces.append(result)
    return detected_faces