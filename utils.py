import os
import cv2
import json
import pyttsx3
import logging
import threading
import ttkbootstrap as tb
from mtcnn import MTCNN
from keras_facenet import FaceNet




embedder = FaceNet()
detector = MTCNN()
engine = pyttsx3.init()
notification_lock = threading.Lock()
engine.setProperty('rate', 150)
attendance_threshold = {"hour": 8}
password_file = "admin_password.json"
dataset_dir = 'dataset1(FaceNet)'
dataset_tunggal = 'dataset_tunggal'
model_path = 'trainer.yml'
csv_file = 'attendance6.csv'
labels_file = 'labels1(FaceNet).txt'
log_file = 'app.log'
admin_file = 'admin_pass.txt'

root = tb.Window(themename="flatly")
root.title("Face Attendance (Extended)")
root.geometry("480x740")
notebook = tb.Notebook(root)
admin_tab = tb.Frame(notebook)
user_tab = tb.Frame(notebook)
notebook.pack(fill='both', expand=True, padx=10, pady=10)
notebook.add(user_tab, text="User")
name_entry_admin = tb.Entry(admin_tab, width=30)
name_entry_admin.pack(pady=5)
name_entry_user = tb.Entry(admin_tab, width=30)   
name_entry_user.pack(pady=5)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

os.makedirs(dataset_dir, exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')


if os.path.exists("attendance_config.json"):
    with open("attendance_config.json", "r") as f:
        try:
            attendance_threshold = json.load(f)
        except json.JSONDecodeError:
            pass 

def password_s(): 
    if os.path.exists(password_file):
        with open(password_file, "r") as f:
            try:
                admin_password = json.load(f).get("password", "admin123")
            except json.JSONDecodeError:
                admin_password = "admin123"
    else:
        admin_password = "admin123"
    return admin_password