import os
import cv2
import numpy as np
from tkinter import messagebox
from attendace import mark_attendance
from utils import dataset_dir
from concurrent.futures import ThreadPoolExecutor
from face_detection import detect_faces_mtcnn, get_batch_face_embeddings

def recognize_faces_facenet_with_threads():
    if not os.path.exists(dataset_dir):
        messagebox.showerror("Error", "Belum ada data.")
        return
    cam = cv2.VideoCapture(0)
    known_embeddings = {}
    for person_name in os.listdir(dataset_dir):
        emb_path = os.path.join(dataset_dir, person_name, f'{person_name}_embedding.npy')
        if os.path.exists(emb_path):
            known_embeddings[person_name] = np.load(emb_path)
    recognized = []
    with ThreadPoolExecutor(max_workers=8) as executor: 
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            futures = []
            future = executor.submit(detect_faces_mtcnn, rgb)
            faces_results = future.result()
            faces = []
            for res in faces_results:
                x, y, w, h = res['box']
                face = frame[y:y+h, x:x+w]
                faces.append(face)
            if faces:
                future_embeddings = executor.submit(get_batch_face_embeddings, faces)
                embeddings = future_embeddings.result()
                for emb, res in zip(embeddings, faces_results):
                    emb = emb / np.linalg.norm(emb)
                    x, y, w, h = res['box']
                    min_dist = float('inf')
                    identity = "Unknown"
                    for name, known_embs in known_embeddings.items():
                        for e in known_embs:
                            dist = np.linalg.norm(emb - e)
                            if dist < min_dist:
                                min_dist = dist
                                identity = name
                    if min_dist < 0.6:  
                        if identity not in recognized:
                            mark_attendance(identity, mode="auto")
                            recognized.append(identity)
                        color = (0, 255, 0)
                        label = f"{identity} ({min_dist:.2f})"
                    elif min_dist < 0.7 :
                        if identity not in recognized:
                            mark_attendance(identity, mode="auto")
                            recognized.append(identity)
                        color = (0, 255, 0)
                        label = f"{identity} ({min_dist:.2f})"
                    elif min_dist < 0.8 : #paling minimal kayaknya
                        if identity not in recognized:
                            mark_attendance(identity, mode="auto")
                            recognized.append(identity)
                        color = (0, 255, 0)
                        label = f"{identity} ({min_dist:.2f})"
                    elif min_dist < 0.9 :
                        if identity not in recognized:
                            mark_attendance(identity, mode="auto")
                            recognized.append(identity)
                        color = (0, 255, 0)
                        label = f"{identity} ({min_dist:.2f})"
                    else:
                        color = (0, 0, 255)
                        label = "Unknown"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Face Recognition - Press 'q' to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cam.release()
    cv2.destroyAllWindows()
