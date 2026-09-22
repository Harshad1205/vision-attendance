import os
import datetime
import urllib.parse
import urllib.request
import numpy as np
import cv2
from bson.binary import Binary
from pymongo import MongoClient, errors
from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="VisionPass Enterprise")

# Ensure required directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================
# MONGODB ATLAS CONFIGURATION
# ============================================================
MONGO_USER = os.getenv("MONGO_USER", "vision_admin")
MONGO_PASS = os.getenv("MONGO_PASS", "vision_123")

ENCODED_USER = urllib.parse.quote_plus(MONGO_USER)
ENCODED_PASS = urllib.parse.quote_plus(MONGO_PASS)

DEFAULT_URI = f"mongodb+srv://{ENCODED_USER}:{ENCODED_PASS}@cluster0.zpn3evs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_URI)

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["visionpass_db"]
    client.admin.command("ping")
    print("[INFO] Connected to MongoDB Atlas successfully.")
except errors.PyMongoError as e:
    print(f"[WARNING] MongoDB Connection failed: {e}")
    db = None

# ============================================================
# HAAR CASCADE FACE DETECTOR LOADER
# ============================================================
CASCADE_FILE = "haarcascade_frontalface_default.xml"

def get_face_cascade():
    """Loads Haar Cascade locally or downloads if missing."""
    if os.path.exists(CASCADE_FILE) and os.path.getsize(CASCADE_FILE) > 50000:
        cascade = cv2.CascadeClassifier(CASCADE_FILE)
        if not cascade.empty():
            return cascade

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        builtin_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
        if os.path.exists(builtin_path):
            cascade = cv2.CascadeClassifier(builtin_path)
            if not cascade.empty():
                return cascade

    url = f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{CASCADE_FILE}"
    try:
        print("[INFO] Fetching Haar Cascade XML from OpenCV repository...")
        urllib.request.urlretrieve(url, CASCADE_FILE)
        cascade = cv2.CascadeClassifier(CASCADE_FILE)
        if not cascade.empty():
            return cascade
    except Exception as exc:
        print(f"[ERROR] Could not load or download Haar Cascade: {exc}")

    raise RuntimeError("Haar Cascade Classifier could not be initialized.")

face_cascade = get_face_cascade()

def extract_face(image_bytes):
    """Crops and normalizes the largest face found in image bytes."""
    if not image_bytes:
        return None, None

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_roi = cv2.resize(gray[y:y + h, x:x + w], (120, 120))
    _, buffer = cv2.imencode(".jpg", face_roi)
    return face_roi, buffer.tobytes()

# ============================================================
# API ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/logs")
def get_logs():
    if db is None:
        return JSONResponse({"status": "error", "message": "Database offline."}, status_code=503)

    try:
        records = list(db.attendance.find().sort("_id", -1).limit(10))
        latest_entry = db.attendance.find_one({"action": "ENTRY"}, sort=[("_id", -1)])
        latest_exit = db.attendance.find_one({"action": "EXIT"}, sort=[("_id", -1)])

        clean_records = [
            {
                "name": r.get("name", "Unknown"),
                "action": r.get("action", "SCAN"),
                "timestamp": r.get("timestamp", "")
            }
            for r in records
        ]

        entry_text = f"{latest_entry['name']} ({latest_entry['timestamp']})" if latest_entry else "None"
        exit_text = f"{latest_exit['name']} ({latest_exit['timestamp']})" if latest_exit else "None"

        return {
            "logs": clean_records,
            "latest_entry": entry_text,
            "latest_exit": exit_text
        }
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)

@app.post("/api/register")
async def register(name: str = Form(""), frame: UploadFile = None):
    if db is None:
        return JSONResponse({"status": "error", "message": "Database not reachable."}, status_code=503)

    clean_name = name.strip()
    if not clean_name:
        return JSONResponse({"status": "error", "message": "Name cannot be empty."}, status_code=400)

    if frame is None:
        return JSONResponse({"status": "error", "message": "No image frame provided."}, status_code=400)

    raw_bytes = await frame.read()
    face_roi, jpg_bytes = extract_face(raw_bytes)
    if face_roi is None:
        return JSONResponse({"status": "error", "message": "No face detected in camera capture."}, status_code=400)

    try:
        highest = db.users.find_one(sort=[("user_id", -1)])
        user_id = (int(highest["user_id"]) + 1) if (highest and "user_id" in highest) else 1
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db.users.update_one(
            {"name": clean_name},
            {"$setOnInsert": {"user_id": user_id, "created_at": now_str}},
            upsert=True
        )

        user_doc = db.users.find_one({"name": clean_name})
        uid = int(user_doc["user_id"])

        db.face_photos.update_one(
            {"user_id": uid},
            {"$set": {"user_id": uid, "name": clean_name, "image_bytes": Binary(jpg_bytes)}},
            upsert=True
        )

        return {"status": "success", "message": f"Successfully registered {clean_name} (ID: {uid})"}
    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"Registration failed: {str(exc)}"}, status_code=500)

@app.post("/api/scan")
async def scan(action: str = Form("ENTRY"), frame: UploadFile = None):
    if db is None:
        return JSONResponse({"status": "error", "message": "Database not reachable."}, status_code=503)

    if frame is None:
        return JSONResponse({"status": "error", "message": "No image frame received."}, status_code=400)

    raw_bytes = await frame.read()
    face_roi, _ = extract_face(raw_bytes)
    if face_roi is None:
        return JSONResponse({"status": "error", "message": "No face detected. Look directly at camera."}, status_code=400)

    try:
        photos = list(db.face_photos.find())
        if not photos:
            return JSONResponse({"status": "error", "message": "No registered users in system."}, status_code=400)

        training_faces = []
        labels = []
        user_map = {}

        for p in photos:
            uid = int(p["user_id"])
            user_map[uid] = p.get("name", f"User {uid}")
            raw_img = np.frombuffer(p["image_bytes"], np.uint8)
            img = cv2.imdecode(raw_img, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                training_faces.append(cv2.resize(img, (120, 120)))
                labels.append(uid)

        if not training_faces or not labels:
            return JSONResponse({"status": "error", "message": "Invalid training data in database."}, status_code=400)

        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            return JSONResponse({"status": "error", "message": "opencv-contrib-python-headless required."}, status_code=500)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(training_faces, np.array(labels, dtype=np.int32))
        pred_id, dist = recognizer.predict(face_roi)

        RECOGNITION_THRESHOLD = 70.0

        if dist <= RECOGNITION_THRESHOLD and pred_id in user_map:
            matched_name = user_map[pred_id]
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            db.attendance.insert_one({
                "user_id": pred_id,
                "name": matched_name,
                "action": action.strip().upper(),
                "confidence_dist": round(float(dist), 2),
                "timestamp": now_str
            })
            return {"status": "success", "name": matched_name, "distance": round(dist, 2)}

        return JSONResponse({
            "status": "error",
            "message": f"Face not recognized (Distance: {int(dist)})."
        }, status_code=401)

    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"Scan processing failed: {str(exc)}"}, status_code=500)