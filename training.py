import os
import cv2
import numpy as np
from utils import face_cascade
from face_detection import get_face_embedding


def train_model_from_folder():
    dataset_tunggal = '/dataset'
    dataset_dir = '/dataset_tes'
    for person_name in os.listdir(dataset_tunggal):
        person_dir = os.path.join(dataset_dir, person_name)
        image_path = os.path.join(dataset_tunggal, person_name, f"{person_name}.jpg")  # contoh: dataset/Tuan/Tuan.jpg

        if not os.path.isfile(image_path):
            continue
        if os.path.exists(person_dir):
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"No face detected in {person_name}")
            continue

        os.makedirs(person_dir, exist_ok=True)

        count = 0
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            for i in range(1, 41):  # hasilkan 40 data
                count += 1
                file_name = f"{person_name}_{count}.jpg"
                cv2.imwrite(os.path.join(person_dir, file_name), face_img)
            break  # hanya proses 1 wajah saja

        # proses embedding
        embeddings = []
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path)
            emb = get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        np.save(os.path.join(person_dir, f'{person_name}_embedding.npy'), embeddings)    
