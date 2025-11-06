import os
import hashlib
from tkinter import messagebox
from utils import dataset_dir, admin_file

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
def is_admin(username, password):
    if not os.path.exists(admin_file):
        return False
    hashed = hash_password(password)
    with open(admin_file, 'r') as f:
        for line in f:
            u, p = line.strip().split(',')
            if u == username and p == hashed:
                return True
    return False
def register_admin(username, password):
    with open(admin_file, 'a') as f:
        f.write(f"{username},{hash_password(password)}\n")
def delete_user(name):
    person_dir = os.path.join(dataset_dir, name)
    if os.path.exists(person_dir):
        for file in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, file))
        os.rmdir(person_dir)
        messagebox.showinfo("Delete", f"Data untuk {name} telah dihapus.")
    else:
        messagebox.showerror("Error", f"Data untuk {name} tidak ditemukan.")                