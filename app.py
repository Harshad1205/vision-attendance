import os
import sys
import csv
import io
import datetime
import urllib.request
import urllib.parse
import numpy as np
import cv2
from PIL import Image
import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
from pymongo import MongoClient, errors
from bson.binary import Binary
# ============================================================
# CONFIGURATION & CREDENTIALS
# ============================================================

APP_NAME = "VisionPass Enterprise | Cloud AI Biometrics"
DATASET_DIR = "dataset"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINING_IMAGES = 30
RECOGNITION_THRESHOLD = 68  # LBPH distance (lower = stricter match)
MIN_FACE_SIZE = (80, 80)
ADMIN_PASSWORD = "admin"

# --- MongoDB Atlas Connection Settings ---
MONGO_USER = os.getenv("MONGO_USER", "vision_admin")
MONGO_PASS = os.getenv("MONGO_PASS", "vision_123")

ENCODED_USER = urllib.parse.quote_plus(MONGO_USER)
ENCODED_PASS = urllib.parse.quote_plus(MONGO_PASS)

DEFAULT_URI = f"mongodb+srv://{ENCODED_USER}:{ENCODED_PASS}@cluster0.zpn3evs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_URI)
DB_NAME = "visionpass_db"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ============================================================
# MAIN APPLICATION
# ============================================================

class VisionPassApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_NAME)
        self.geometry("1300x820")
        self.minsize(1150, 750)
        self.resizable(True, True)

        # Runtime & Model Variables
        self.is_running = True
        self.after_id = None
        self.recognized_name = None
        self.recognized_id = None
        self.recognizer = None
        self.model_ready = False
        self.cap = None

        # User map: {numeric_id: person_name}
        self.user_map = {}

        # MongoDB Initialization
        self.mongo_client = None
        self.db = None
        self.init_mongo()

        os.makedirs(DATASET_DIR, exist_ok=True)
        self.face_cascade = self.load_cascade()
        self.load_recognition_model()
        self.load_user_mappings()
        self.init_treeview_styles()

        # Build Main Frame Container
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)

        # Launch Live Attendance View
        self.show_kiosk_view()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ========================================================
    # DATABASE & STYLES
    # ========================================================

    def init_treeview_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#161B22",
            foreground="#E6EDF3",
            fieldbackground="#161B22",
            rowheight=28,
            bordercolor="#30363D"
        )
        style.configure(
            "Treeview.Heading",
            background="#21262D",
            foreground="#E6EDF3",
            font=("Segoe UI", 10, "bold")
        )
        style.map("Treeview", background=[("selected", "#1F6FEB")])

    def init_mongo(self):
        try:
            self.mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client[DB_NAME]
            self.mongo_client.admin.command('ping')
            
            self.db.users.create_index("user_id", unique=True)
            self.db.users.create_index("name", unique=True)
            print("[INFO] Connected to MongoDB Atlas successfully.")
        except errors.PyMongoError as e:
            print(f"[ERROR] MongoDB Connection Error: {e}")
            messagebox.showwarning(
                "Database Warning",
                "Could not connect to MongoDB Atlas.\nCheck network access whitelist or credentials."
            )
            self.db = None

    def load_user_mappings(self):
        if self.db is None:
            return
        try:
            cursor = self.db.users.find({}, {"user_id": 1, "name": 1})
            self.user_map = {int(doc["user_id"]): doc["name"] for doc in cursor if "user_id" in doc}
        except Exception as e:
            print(f"[ERROR] Failed to load users from MongoDB: {e}")

    def clear_container(self):
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        for widget in self.container.winfo_children():
            widget.destroy()

    # ========================================================
    # CAMERA & MODEL HELPERS
    # ========================================================

    def init_camera(self):
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)

        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def load_cascade(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_dir, CASCADE_FILE)

        if not os.path.exists(local_path) or os.path.getsize(local_path) < 50000:
            url = f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{CASCADE_FILE}"
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"[ERROR] Could not download cascade: {e}")

        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            return cascade

        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            builtin_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
            if os.path.exists(builtin_path):
                cascade = cv2.CascadeClassifier(builtin_path)
                if not cascade.empty():
                    return cascade

        raise FileNotFoundError("Could not load Haar Cascade XML.")

    def load_recognition_model(self):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            self.model_ready = False
            return

        model_path = os.path.join(DATASET_DIR, "trainer.yml")
        if not os.path.exists(model_path):
            self.model_ready = False
            return

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(model_path)
            self.model_ready = True
        except Exception as e:
            self.model_ready = False
            print(f"[ERROR] Loading trainer.yml: {e}")

    # ========================================================
    # VIEW 1: LIVE ATTENDANCE KIOSK
    # ========================================================

    def show_kiosk_view(self):
        self.clear_container()

        self.container.grid_columnconfigure(0, weight=3)
        self.container.grid_columnconfigure(1, weight=2)
        self.container.grid_rowconfigure(0, weight=1)

        # ----------------- LEFT PANEL: CAMERA & LIVE STATUS -----------------
        left_panel = ctk.CTkFrame(self.container, corner_radius=12)
        left_panel.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="nsew")

        self.video_label = ctk.CTkLabel(left_panel, text="Initializing Camera Stream...")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=(10, 5))

        # Bottom Live Cards: Persistent Last Entry and Last Exit
        live_cards_frame = ctk.CTkFrame(left_panel, fg_color="#1F242C", corner_radius=10)
        live_cards_frame.pack(fill="x", padx=10, pady=(5, 10))
        live_cards_frame.grid_columnconfigure((0, 1), weight=1)

        # Card: Last Entry
        entry_card = ctk.CTkFrame(live_cards_frame, fg_color="#161B22", corner_radius=8, border_width=1, border_color="#2EA043")
        entry_card.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        ctk.CTkLabel(entry_card, text="LATEST ENTRY", font=ctk.CTkFont(size=11, weight="bold"), text_color="#2EA043").pack(anchor="w", padx=10, pady=(6, 2))
        
        entry_content = ctk.CTkFrame(entry_card, fg_color="transparent")
        entry_content.pack(fill="x", padx=8, pady=4)
        
        self.entry_photo_lbl = ctk.CTkLabel(entry_content, text="[No Photo]", width=65, height=65, fg_color="#0D1117", corner_radius=6)
        self.entry_photo_lbl.pack(side="left", padx=5)

        self.entry_info_lbl = ctk.CTkLabel(entry_content, text="Name: None\nTime: --:--:--", justify="left", font=ctk.CTkFont(size=12))
        self.entry_info_lbl.pack(side="left", padx=8)

        # Card: Last Exit
        exit_card = ctk.CTkFrame(live_cards_frame, fg_color="#161B22", corner_radius=8, border_width=1, border_color="#DA3633")
        exit_card.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")

        ctk.CTkLabel(exit_card, text="LATEST EXIT", font=ctk.CTkFont(size=11, weight="bold"), text_color="#DA3633").pack(anchor="w", padx=10, pady=(6, 2))
        
        exit_content = ctk.CTkFrame(exit_card, fg_color="transparent")
        exit_content.pack(fill="x", padx=8, pady=4)
        
        self.exit_photo_lbl = ctk.CTkLabel(exit_content, text="[No Photo]", width=65, height=65, fg_color="#0D1117", corner_radius=6)
        self.exit_photo_lbl.pack(side="left", padx=5)

        self.exit_info_lbl = ctk.CTkLabel(exit_content, text="Name: None\nTime: --:--:--", justify="left", font=ctk.CTkFont(size=12))
        self.exit_info_lbl.pack(side="left", padx=8)

        # ----------------- RIGHT PANEL: ACTIONS & CLOUD LOGS -----------------
        self.control_frame = ctk.CTkFrame(self.container, corner_radius=12)
        self.control_frame.grid(row=0, column=1, padx=(5, 15), pady=15, sticky="nsew")

        # Top Bar
        header_bar = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        header_bar.pack(fill="x", padx=15, pady=(15, 5))

        ctk.CTkLabel(header_bar, text="VisionPass Kiosk", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        btn_admin = ctk.CTkButton(
            header_bar,
            text="🔒 Admin Panel",
            width=110,
            height=30,
            fg_color="#21262D",
            hover_color="#30363D",
            command=self.prompt_admin_login
        )
        btn_admin.pack(side="right")

        self.status_label = ctk.CTkLabel(
            self.control_frame,
            text="Status: Ready (Cloud Connected)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3B8ED0"
        )
        self.status_label.pack(pady=2)

        self.recognition_label = ctk.CTkLabel(
            self.control_frame,
            text="Recognition: Waiting...",
            font=ctk.CTkFont(size=13)
        )
        self.recognition_label.pack(pady=(0, 6))

        # Registration Panel
        reg_box = ctk.CTkFrame(self.control_frame, fg_color="#1F242C", corner_radius=8)
        reg_box.pack(fill="x", padx=15, pady=4)

        ctk.CTkLabel(reg_box, text="Register New Face to MongoDB", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=12, pady=(6, 2))

        self.name_entry = ctk.CTkEntry(reg_box, placeholder_text="Enter Full Name...", height=34)
        self.name_entry.pack(fill="x", padx=10, pady=3)

        self.register_btn = ctk.CTkButton(
            reg_box,
            text="📷 Register & Upload Face",
            fg_color="#8957E5",
            hover_color="#7042C2",
            height=34,
            command=self.register_face
        )
        self.register_btn.pack(fill="x", padx=10, pady=(3, 8))

        # Train & Action Buttons
        self.train_btn = ctk.CTkButton(
            self.control_frame,
            text="⚙ Train Recognition Model",
            height=34,
            command=self.train_model
        )
        self.train_btn.pack(fill="x", padx=15, pady=4)

        btn_row = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=4)

        self.scan_in_btn = ctk.CTkButton(
            btn_row,
            text="➔ Scan Entry",
            fg_color="#2EA043",
            hover_color="#238636",
            height=36,
            command=lambda: self.log_attendance("ENTRY")
        )
        self.scan_in_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.scan_out_btn = ctk.CTkButton(
            btn_row,
            text="➔ Scan Exit",
            fg_color="#DA3633",
            hover_color="#B62324",
            height=36,
            command=lambda: self.log_attendance("EXIT")
        )
        self.scan_out_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # Logs Section
        ctk.CTkLabel(self.control_frame, text="Live Attendance Feed", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 4))

        tbl_container = ctk.CTkFrame(self.control_frame, corner_radius=8)
        tbl_container.pack(fill="both", expand=True, padx=15, pady=(2, 15))

        self.kiosk_tree = ttk.Treeview(
            tbl_container,
            columns=("name", "action", "time"),
            show="headings",
            selectmode="browse"
        )
        self.kiosk_tree.heading("name", text="Name")
        self.kiosk_tree.heading("action", text="Action")
        self.kiosk_tree.heading("time", text="Timestamp")

        self.kiosk_tree.column("name", width=130, anchor="w")
        self.kiosk_tree.column("action", width=80, anchor="center")
        self.kiosk_tree.column("time", width=140, anchor="center")

        scroll = ttk.Scrollbar(tbl_container, orient="vertical", command=self.kiosk_tree.yview)
        self.kiosk_tree.configure(yscrollcommand=scroll.set)

        self.kiosk_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Initialize Camera Stream & Hydrate Logs
        self.cap = self.init_camera()
        if self.cap is None or not self.cap.isOpened():
            self.video_label.configure(text="Webcam Error.\nCheck camera permissions.", image=None)
            self.status_label.configure(text="Status: Camera Error", text_color="#E5534B")
        else:
            self.update_video_stream()

        self.refresh_kiosk_logs()
        self.update_live_cards()

    def update_video_stream(self):
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=MIN_FACE_SIZE)

            detected_person = None
            detected_id = None
            detected_dist = 999

            for (x, y, w, h) in faces:
                box_color = (0, 0, 255)
                label_text = "Unknown"

                if self.model_ready and self.recognizer is not None:
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    user_id, dist = self.recognizer.predict(face_roi)

                    if dist <= RECOGNITION_THRESHOLD and int(user_id) in self.user_map:
                        detected_person = self.user_map[int(user_id)]
                        detected_id = int(user_id)
                        detected_dist = int(dist)
                        box_color = (0, 255, 0)
                        label_text = f"{detected_person} ({detected_dist})"
                    else:
                        label_text = f"Unknown ({int(dist)})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if detected_person:
                self.recognized_name = detected_person
                self.recognized_id = detected_id
                self.recognition_label.configure(
                    text=f"Recognized: {detected_person} (Dist: {detected_dist})",
                    text_color="#57AB5A"
                )
            else:
                self.recognized_name = None
                self.recognized_id = None
                self.recognition_label.configure(text="Recognition: Waiting...", text_color="#8B949E")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(600, 430))
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image

        self.after_id = self.after(25, self.update_video_stream)

    def register_face(self):
        new_name = self.name_entry.get().strip()
        if not new_name:
            self.status_label.configure(text="Enter a name before registering.", text_color="#E5534B")
            return

        if self.cap is None or not self.cap.isOpened():
            self.status_label.configure(text="Camera unavailable.", text_color="#E5534B")
            return

        existing_user = self.db.users.find_one({"name": new_name}) if self.db is not None else None
        if existing_user:
            user_id = int(existing_user["user_id"])
        else:
            highest_user = self.db.users.find_one(sort=[("user_id", -1)]) if self.db is not None else None
            user_id = (int(highest_user["user_id"]) + 1) if (highest_user and "user_id" in highest_user) else 1
            now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.db is not None:
                self.db.users.insert_one({"user_id": user_id, "name": new_name, "created_at": now_iso})

        self.user_map[user_id] = new_name
        user_folder = os.path.join(DATASET_DIR, f"user_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        self.status_label.configure(text=f"Capturing face for {new_name}...", text_color="#D29922")
        self.register_btn.configure(state="disabled")
        self.scan_in_btn.configure(state="disabled")
        self.scan_out_btn.configure(state="disabled")

        captured = 0
        preview_binary = None

        while captured < TRAINING_IMAGES and self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 6, minSize=MIN_FACE_SIZE)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                captured += 1

                img_path = os.path.join(user_folder, f"face_{captured}.jpg")
                cv2.imwrite(img_path, face)

                # Save snapshot as binary for database storage
                if captured == 1:
                    _, buffer = cv2.imencode('.jpg', face)
                    preview_binary = Binary(buffer.tobytes())

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Capturing: {captured}/{TRAINING_IMAGES}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show exactly ONE face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(600, 430))
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image
            self.update()

        # Save photo record to MongoDB
        if self.db is not None and preview_binary is not None:
            self.db.face_photos.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "name": new_name, "image_bytes": preview_binary}},
                upsert=True
            )

        self.register_btn.configure(state="normal")
        self.scan_in_btn.configure(state="normal")
        self.scan_out_btn.configure(state="normal")
        self.name_entry.delete(0, 'end')

        self.status_label.configure(text=f"Uploaded {new_name} & updated model!", text_color="#57AB5A")
        self.train_model()

    def train_model(self):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            self.status_label.configure(text="Install opencv-contrib-python!", text_color="#E5534B")
            return

        self.load_user_mappings()
        faces = []
        labels = []

        for user_id in self.user_map.keys():
            user_dir = os.path.join(DATASET_DIR, f"user_{user_id}")
            if not os.path.exists(user_dir):
                continue
            for file in os.listdir(user_dir):
                if file.lower().endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(user_dir, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(int(user_id))

        if len(faces) == 0:
            self.status_label.configure(text="No images to train. Register a face first.", text_color="#E5534B")
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            # Explicit contiguous 32-bit integer array to prevent native memory exceptions
            recognizer.train(faces, np.array(labels, dtype=np.int32))
            recognizer.write(os.path.join(DATASET_DIR, "trainer.yml"))
            self.recognizer = recognizer
            self.model_ready = True
            self.status_label.configure(text="Model active & up to date!", text_color="#57AB5A")
        except Exception as e:
            self.status_label.configure(text=f"Training Failed: {e}", text_color="#E5534B")

    def log_attendance(self, action):
        if not self.recognized_name or not self.recognized_id:
            self.status_label.configure(text="Face not identified yet.", text_color="#E5534B")
            return

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save record to MongoDB
        if self.db is not None:
            self.db.attendance.insert_one({
                "user_id": int(self.recognized_id),
                "name": self.recognized_name,
                "action": action,
                "timestamp": now_str
            })

        self.status_label.configure(text=f"{action} logged for {self.recognized_name}", text_color="#57AB5A")
        self.refresh_kiosk_logs()
        self.update_live_cards()

    def refresh_kiosk_logs(self):
        for item in self.kiosk_tree.get_children():
            self.kiosk_tree.delete(item)

        if self.db is None:
            return

        try:
            records = self.db.attendance.find().sort("_id", -1).limit(25)
            for r in records:
                self.kiosk_tree.insert("", "end", values=(r.get("name"), r.get("action"), r.get("timestamp")))
        except Exception as e:
            print(f"[ERROR] Failed to fetch kiosk logs: {e}")

    def update_live_cards(self):
        if self.db is None:
            return

        # Fetch latest ENTRY
        latest_entry = self.db.attendance.find_one({"action": "ENTRY"}, sort=[("_id", -1)])
        if latest_entry:
            self.entry_info_lbl.configure(text=f"Name: {latest_entry.get('name')}\nTime: {latest_entry.get('timestamp')}")
            photo_doc = self.db.face_photos.find_one({"user_id": int(latest_entry.get("user_id"))})
            if photo_doc and "image_bytes" in photo_doc:
                pil_img = Image.open(io.BytesIO(photo_doc["image_bytes"]))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(65, 65))
                self.entry_photo_lbl.configure(image=ctk_img, text="")
                self.entry_photo_lbl.image = ctk_img

        # Fetch latest EXIT
        latest_exit = self.db.attendance.find_one({"action": "EXIT"}, sort=[("_id", -1)])
        if latest_exit:
            self.exit_info_lbl.configure(text=f"Name: {latest_exit.get('name')}\nTime: {latest_exit.get('timestamp')}")
            photo_doc = self.db.face_photos.find_one({"user_id": int(latest_exit.get("user_id"))})
            if photo_doc and "image_bytes" in photo_doc:
                pil_img = Image.open(io.BytesIO(photo_doc["image_bytes"]))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(65, 65))
                self.exit_photo_lbl.configure(image=ctk_img, text="")
                self.exit_photo_lbl.image = ctk_img

    # ========================================================
    # VIEW 2: ADMIN PANEL WITH PHOTO INSPECTOR
    # ========================================================

    def prompt_admin_login(self):
        dialog = ctk.CTkInputDialog(text="Enter Admin Password:", title="Admin Verification")
        pwd = dialog.get_input()
        if pwd == ADMIN_PASSWORD:
            self.show_admin_view()
        elif pwd is not None:
            messagebox.showerror("Denied", "Incorrect Admin Password.")

    def show_admin_view(self):
        self.clear_container()

        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=0)

        # Header Navigation Bar
        top_bar = ctk.CTkFrame(self.container, height=60, corner_radius=0)
        top_bar.pack(fill="x", side="top", padx=0, pady=0)

        ctk.CTkLabel(
            top_bar,
            text="VisionPass | Cloud Admin Portal (MongoDB Atlas)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left", padx=20, pady=15)

        btn_back = ctk.CTkButton(
            top_bar,
            text="← Back to Kiosk",
            width=120,
            height=32,
            fg_color="#21262D",
            hover_color="#30363D",
            command=self.show_kiosk_view
        )
        btn_back.pack(side="right", padx=20, pady=15)

        tabs = ctk.CTkTabview(self.container)
        tabs.pack(fill="both", expand=True, padx=20, pady=15)

        tab_logs = tabs.add("Attendance Records")
        tab_users = tabs.add("Registered Users & Photos")

        # --- TAB 1: ATTENDANCE RECORDS ---
        log_actions = ctk.CTkFrame(tab_logs, fg_color="transparent")
        log_actions.pack(fill="x", pady=(5, 10))

        btn_export = ctk.CTkButton(
            log_actions,
            text="📥 Export to CSV",
            fg_color="#238636",
            hover_color="#2EA043",
            command=self.export_logs_csv
        )
        btn_export.pack(side="left", padx=5)

        btn_refresh = ctk.CTkButton(
            log_actions,
            text="🔄 Refresh MongoDB",
            fg_color="#1F6FEB",
            hover_color="#388BFD",
            command=lambda: self.load_admin_logs(admin_tree)
        )
        btn_refresh.pack(side="left", padx=5)

        admin_tree = ttk.Treeview(
            tab_logs,
            columns=("id", "name", "action", "timestamp"),
            show="headings"
        )
        admin_tree.heading("id", text="Doc ID")
        admin_tree.heading("name", text="User Name")
        admin_tree.heading("action", text="Action")
        admin_tree.heading("timestamp", text="Timestamp")

        admin_tree.column("id", width=220, anchor="center")
        admin_tree.column("name", width=200, anchor="w")
        admin_tree.column("action", width=120, anchor="center")
        admin_tree.column("timestamp", width=200, anchor="center")

        admin_tree.pack(fill="both", expand=True, pady=5)
        self.load_admin_logs(admin_tree)

        # --- TAB 2: REGISTERED USERS & PHOTOS ---
        users_split = ctk.CTkFrame(tab_users, fg_color="transparent")
        users_split.pack(fill="both", expand=True, pady=10)

        # Left side: Treeview
        users_tree = ttk.Treeview(
            users_split,
            columns=("uid", "name", "created"),
            show="headings",
            selectmode="browse"
        )
        users_tree.heading("uid", text="User Numeric ID")
        users_tree.heading("name", text="Full Name")
        users_tree.heading("created", text="Created On")

        users_tree.column("uid", width=120, anchor="center")
        users_tree.column("name", width=220, anchor="w")
        users_tree.column("created", width=180, anchor="center")

        users_tree.pack(side="left", fill="both", expand=True, padx=(0, 15))

        # Right side: Registration Photo Inspector
        photo_panel = ctk.CTkFrame(users_split, width=280, fg_color="#161B22", corner_radius=10)
        photo_panel.pack(side="right", fill="y", padx=5, pady=5)
        photo_panel.pack_propagate(False)

        ctk.CTkLabel(photo_panel, text="Cloud Stored Photo", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10))

        self.admin_photo_lbl = ctk.CTkLabel(photo_panel, text="Select a user\nto inspect face", width=180, height=180, fg_color="#0D1117", corner_radius=8)
        self.admin_photo_lbl.pack(pady=10)

        self.admin_photo_name = ctk.CTkLabel(photo_panel, text="", font=ctk.CTkFont(size=13, weight="bold"))
        self.admin_photo_name.pack(pady=5)

        def on_user_select(event):
            selected = users_tree.selection()
            if not selected:
                return
            item = users_tree.item(selected[0])
            user_id = item['values'][0]
            name = item['values'][1]

            self.admin_photo_name.configure(text=f"{name} (ID: {user_id})")

            # Fetch image from MongoDB face_photos collection
            if self.db is not None:
                doc = self.db.face_photos.find_one({"user_id": int(user_id)})
                if doc and "image_bytes" in doc:
                    pil_img = Image.open(io.BytesIO(doc["image_bytes"]))
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(180, 180))
                    self.admin_photo_lbl.configure(image=ctk_img, text="")
                    self.admin_photo_lbl.image = ctk_img
                else:
                    self.admin_photo_lbl.configure(image=None, text="No Cloud Photo")

        users_tree.bind("<<TreeviewSelect>>", on_user_select)
        self.load_admin_users(users_tree)

    def load_admin_logs(self, tree):
        for item in tree.get_children():
            tree.delete(item)
        if self.db is None:
            return
        records = self.db.attendance.find().sort("_id", -1).limit(500)
        for r in records:
            tree.insert("", "end", values=(str(r.get("_id")), r.get("name"), r.get("action"), r.get("timestamp")))

    def load_admin_users(self, tree):
        for item in tree.get_children():
            tree.delete(item)
        if self.db is None:
            return
        users = self.db.users.find().sort("user_id", 1)
        for u in users:
            tree.insert("", "end", values=(u.get("user_id"), u.get("name"), u.get("created_at")))

    def export_logs_csv(self):
        if self.db is None:
            messagebox.showerror("Error", "No active MongoDB connection.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Attendance Logs"
        )
        if not file_path:
            return

        records = list(self.db.attendance.find().sort("_id", -1))
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["MongoDB_ID", "Name", "Action", "Timestamp"])
            for r in records:
                writer.writerow([str(r.get("_id")), r.get("name"), r.get("action"), r.get("timestamp")])

        messagebox.showinfo("Export Successful", f"Saved {len(records)} records to:\n{file_path}")

    # ========================================================
    # CLEANUP
    # ========================================================

    def on_close(self):
        self.is_running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.mongo_client:
            self.mongo_client.close()
        self.destroy()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = VisionPassApp()
    app.mainloop()