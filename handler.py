import os
import json
import pandas as pd
import tkinter as tk
import ttkbootstrap as tb
from admin import delete_user
from tkinter import messagebox, ttk
from ttkbootstrap.constants import *
from face_detection import register_face
from attendace import show_today_attendance
from training import train_model_from_folder
from recognizing import recognize_faces_facenet_with_threads
from utils import root, name_entry_user, name_entry_admin, dataset_dir, dataset_tunggal, csv_file, password_file, notebook, admin_tab, password_s

def on_register():
        name = name_entry_user.get().strip()
        if name:
            register_face(name)
        else:
            messagebox.showerror("Error", "Isi nama dulu!")
def on_train():
        train_model_from_folder(dataset_tunggal, dataset_dir)
        messagebox.showinfo("Train", "Model training selesai!")
def on_recognize():
        recognize_faces_facenet_with_threads()
def on_delete():
    name = name_entry_admin.get().strip()
    if name:
        delete_user(name)
    else:
        messagebox.showerror("Error", "Masukkan nama untuk dihapus!")
def on_today():
    show_today_attendance()
def set_attendance_time():
    global attendance_threshold
    def save_time():
        try:
            pagi_hour = int(pagi_hour_combobox.get())
            pagi_minute = int(pagi_minute_combobox.get())
            siang_hour = int(siang_hour_combobox.get())
            siang_minute = int(siang_minute_combobox.get())
            malam_hour = int(malam_hour_combobox.get())
            malam_minute = int(malam_minute_combobox.get())
            if 0 <= pagi_hour <= 23 and 0 <= pagi_minute <= 59 and \
                0 <= siang_hour <= 23 and 0 <= siang_minute <= 59 and \
                0 <= malam_hour <= 23 and 0 <= malam_minute <= 59:
                    attendance_threshold["pagi"] = {"hour": pagi_hour, "minute": pagi_minute}
                    attendance_threshold["siang"] = {"hour": siang_hour, "minute": siang_minute}
                    attendance_threshold["malam"] = {"hour": malam_hour, "minute": malam_minute}
                    max_shift_hour = max(malam_hour, siang_hour, pagi_hour)
                    max_shift_minute = max(malam_minute, siang_minute, pagi_minute)
                    attendance_threshold["shift_start_time"] = {
                        "hour": (max_shift_hour - 1) % 24,
                        "minute": max_shift_minute
                    }
                    with open("attendance_config.json", "w") as f:
                        json.dump(attendance_threshold, f)
                    messagebox.showinfo("Berhasil", f"Pengaturan shift telah diset.")
                    config_window.destroy()
            else:
                messagebox.showerror("Error", "Jam, menit tidak valid.")
        except ValueError:
            messagebox.showerror("Error", "Pilih jam, menit yang valid.")
        config_window = tb.Toplevel(root)
        config_window.title("Pengaturan Shift dan Jam Masuk")
        config_window.geometry("400x350")
        frame_main = tb.Frame(config_window, bootstyle="light")
        frame_main.pack(padx=20, pady=20, fill="both", expand=True)
        tb.Label(frame_main, text="Pengaturan Shift", font=("Segoe UI", 16, "bold"), bootstyle="info").pack(pady=10)
        tb.Label(frame_main, text="Pilih jam masuk shift pagi:", font=("Segoe UI", 12), bootstyle="secondary").pack(pady=5)
        pagi_hours = [str(i).zfill(2) for i in range(24)]
        pagi_hour_combobox = ttk.Combobox(frame_main, values=pagi_hours, state="readonly", font=("Segoe UI", 14))
        pagi_hour_combobox.set(str(attendance_threshold.get("pagi", {}).get("hour", 8)).zfill(2))
        pagi_hour_combobox.pack(pady=5, ipadx=5, ipady=5)
        pagi_minutes = [str(i).zfill(2) for i in range(60)]
        pagi_minute_combobox = ttk.Combobox(frame_main, values=pagi_minutes, state="readonly", font=("Segoe UI", 14))
        pagi_minute_combobox.set(str(attendance_threshold.get("pagi", {}).get("minute", 0)).zfill(2))
        pagi_minute_combobox.pack(pady=5, ipadx=5, ipady=5)
        tb.Label(frame_main, text="Pilih jam masuk shift siang:", font=("Segoe UI", 12), bootstyle="secondary").pack(pady=5)
        siang_hour_combobox = ttk.Combobox(frame_main, values=pagi_hours, state="readonly", font=("Segoe UI", 14))
        siang_hour_combobox.set(str(attendance_threshold.get("siang", {}).get("hour", 14)).zfill(2))
        siang_hour_combobox.pack(pady=5, ipadx=5, ipady=5)
        siang_minute_combobox = ttk.Combobox(frame_main, values=pagi_minutes, state="readonly", font=("Segoe UI", 14))
        siang_minute_combobox.set(str(attendance_threshold.get("siang", {}).get("minute", 0)).zfill(2))
        siang_minute_combobox.pack(pady=5, ipadx=5, ipady=5)
        tb.Label(frame_main, text="Pilih jam masuk shift malam:", font=("Segoe UI", 12), bootstyle="secondary").pack(pady=5)
        malam_hour_combobox = ttk.Combobox(frame_main, values=pagi_hours, state="readonly", font=("Segoe UI", 14))
        malam_hour_combobox.set(str(attendance_threshold.get("malam", {}).get("hour", 22)).zfill(2))
        malam_hour_combobox.pack(pady=5, ipadx=5, ipady=5)
        malam_minute_combobox = ttk.Combobox(frame_main, values=pagi_minutes, state="readonly", font=("Segoe UI", 14))
        malam_minute_combobox.set(str(attendance_threshold.get("malam", {}).get("minute", 0)).zfill(2))
        malam_minute_combobox.pack(pady=5, ipadx=5, ipady=5)
        tb.Button(frame_main, text="Simpan", command=save_time, bootstyle="success", width=20).pack(pady=10)
        tb.Button(frame_main, text="Batal", command=config_window.destroy, bootstyle="danger", width=20).pack(pady=5)
def show_all_attendance():
        if not os.path.exists(csv_file):
            messagebox.showerror("Error", "File absensi tidak ditemukan.")
            return
        df = pd.read_csv(csv_file)
        if df.empty:
            messagebox.showinfo("Absensi", "Data absensi masih kosong.")
            return
        top = tk.Toplevel(root)
        top.title("Semua Data Absensi")
        top.geometry("1000x550")
        top.configure(bg="#f0f0f0")
        title = ttk.Label(top, text="Data Absensi Keseluruhan", font=('Segoe UI', 14, 'bold'))
        search_frame = ttk.Frame(top)
        search_frame.pack(pady=5)
        search_label = ttk.Label(search_frame, text="Cari Nama atau Tanggal:")
        search_label.pack(side="left", padx=(0, 5))
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30)
        search_entry.pack(side="left")
        def filter_data():
            keyword = search_var.get().lower().strip()
            filtered_df = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(keyword).any(), axis=1)]
            for item in tree.get_children():
                tree.delete(item)
            for i, row in filtered_df.iterrows():
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                tree.insert('', 'end', values=list(row), tags=(tag,))
        search_btn = ttk.Button(search_frame, text="Cari", command=filter_data)
        search_btn.pack(side="left", padx=5)
        reset_btn = ttk.Button(search_frame, text="Reset", command=lambda: [search_var.set(""), populate_tree(df)])
        reset_btn.pack(side="left")
        frame = ttk.Frame(top)
        frame.pack(fill="both", expand=True, padx=15, pady=10)
        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
        tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        for col in df.columns:
            tree.heading(col, text=col, anchor='center')
            tree.column(col, anchor='center', width=120, stretch=True)
        def populate_tree(dataframe):
            for item in tree.get_children():
                tree.delete(item)
            for i, row in dataframe.iterrows():
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                tree.insert('', 'end', values=list(row), tags=(tag,))
        populate_tree(df)
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Segoe UI', 10, 'bold'))
        style.configure("Treeview", font=('Segoe UI', 10), rowheight=25)
        style.map("Treeview", background=[("selected", "#d1e7dd")])
        tree.tag_configure('evenrow', background='#ffffff')
        tree.tag_configure('oddrow', background='#f5f5f5')
def switch_to_admin():
    def verify_password():
        entered_password = password_entry.get()
        if entered_password == password_s:
            messagebox.showinfo("Akses Diberikan", "Berhasil masuk sebagai admin.")
            notebook.add(admin_tab, text="Admin")
            notebook.select(admin_tab)
            pw_window.destroy()
        else:
            messagebox.showerror("Akses Ditolak", "Password salah!")
    pw_window = tb.Toplevel(root)
    pw_window.title("Verifikasi Admin")
    pw_window.geometry("300x150")
    tb.Label(pw_window, text="Masukkan Password Admin:", bootstyle="info").pack(pady=10)
    password_entry = tb.Entry(pw_window, show="*", width=25)
    password_entry.pack(pady=5)
    tb.Button(pw_window, text="Masuk", command=verify_password, bootstyle="primary").pack(pady=10)
    password_entry.focus()
def change_admin_password():
    def verify_old_password():
        if old_pass_entry.get() == password_s:
            old_pass_window.destroy()
            enter_new_password()
        else:
            messagebox.showerror("Error", "Password lama salah!")
    def enter_new_password():
        def save_new_password():
            new_pass = new_pass_entry.get()
            if len(new_pass) < 4:
                messagebox.showerror("Error", "Password terlalu pendek (min. 4 karakter)")
                return
            with open(password_file, "w") as f:
                json.dump({"password": new_pass}, f)
            messagebox.showinfo("Berhasil", "Password admin berhasil diganti!")
            new_pass_window.destroy()
            global password_s
            password_s = new_pass
        new_pass_window = tb.Toplevel(root)
        new_pass_window.title("Password Baru")
        new_pass_window.geometry("300x150")
        tb.Label(new_pass_window, text="Masukkan Password Baru:", bootstyle="info").pack(pady=10)
        new_pass_entry = tb.Entry(new_pass_window, show="*", width=25)
        new_pass_entry.pack(pady=5)
        tb.Button(new_pass_window, text="Simpan", command=save_new_password, bootstyle="success").pack(pady=10)
    old_pass_window = tb.Toplevel(root)
    old_pass_window.title("Verifikasi Password Lama")
    old_pass_window.geometry("300x150")
    tb.Label(old_pass_window, text="Masukkan Password Lama:", bootstyle="warning").pack(pady=10)
    old_pass_entry = tb.Entry(old_pass_window, show="*", width=25)
    old_pass_entry.pack(pady=5)
    tb.Button(old_pass_window, text="Lanjut", command=verify_old_password, bootstyle="primary").pack(pady=10)






    
    