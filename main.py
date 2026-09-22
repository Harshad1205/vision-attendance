import os
import io
import datetime
import urllib.parse
import numpy as np
import cv2
from PIL import Image
from bson.binary import Binary
from pymongo import MongoClient
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="VisionPass Enterprise")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MongoDB Credentials
MONGO_USER = "vision_admin"
MONGO_PASS = "vision_123"
ENCODED_USER = urllib.parse.quote_plus(MONGO_USER)
ENCODED_PASS = urllib.parse.quote_plus(MONGO_PASS)

MONGO_URI = os.getenv(
    "MONGO_URI",
    f"mongodb+srv://{ENCODED_USER}:{ENCODED_PASS}@cluster0.zpn3evs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

client = MongoClient(MONGO_URI)
db = client["visionpass_db"]

# Load Cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def extract_face(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
    _, buffer = cv2.imencode('.jpg', face_roi)
    return face_roi, buffer.tobytes()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/logs")
def get_logs():
    records = list(db.attendance.find().sort("_id", -1).limit(10))
    latest_entry = db.attendance.find_one({"action": "ENTRY"}, sort=[("_id", -1)])
    latest_exit = db.attendance.find_one({"action": "EXIT"}, sort=[("_id", -1)])
    
    clean_records = [
        {"name": r["name"], "action": r["action"], "timestamp": r["timestamp"]}
        for r in records
    ]
    return {
        "logs": clean_records,
        "latest_entry": latest_entry["name"] + " (" + latest_entry["timestamp"] + ")" if latest_entry else "None",
        "latest_exit": latest_exit["name"] + " (" + latest_exit["timestamp"] + ")" if latest_exit else "None"
    }

@app.post("/api/register")
async def register(name: str = Form(...), frame: UploadFile = File(...)):
    raw = await frame.read()
    face_roi, jpg_bytes = extract_face(raw)
    if face_roi is None:
        return JSONResponse({"status": "error", "message": "No face detected."}, status_code=400)
    
    highest = db.users.find_one(sort=[("user_id", -1)])
    user_id = (highest["user_id"] + 1) if (highest and "user_id" in highest) else 1
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.users.update_one(
        {"name": name},
        {"$setOnInsert": {"user_id": user_id, "created_at": now_str}},
        upsert=True
    )
    user_doc = db.users.find_one({"name": name})
    uid = user_doc["user_id"]

    db.face_photos.update_one(
        {"user_id": uid},
        {"$set": {"user_id": uid, "name": name, "image_bytes": Binary(jpg_bytes)}},
        upsert=True
    )
    return {"status": "success", "message": f"Registered {name} (ID: {uid})"}

@app.post("/api/scan")
async def scan(action: str = Form(...), frame: UploadFile = File(...)):
    raw = await frame.read()
    face_roi, _ = extract_face(raw)
    if face_roi is None:
        return JSONResponse({"status": "error", "message": "No face found in camera feed."}, status_code=400)

    # LBPH training on stored photos
    photos = list(db.face_photos.find())
    if not photos:
        return JSONResponse({"status": "error", "message": "No registered users in database."}, status_code=400)

    training_faces = []
    labels = []
    user_map = {}

    for p in photos:
        uid = p["user_id"]
        user_map[uid] = p["name"]
        arr = np.frombuffer(p["image_bytes"], np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            training_faces.append(cv2.resize(img, (100, 100)))
            labels.append(uid)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(labels))
    pred_id, dist = recognizer.predict(face_roi)

    if dist <= 70 and pred_id in user_map:
        matched_name = user_map[pred_id]
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.attendance.insert_one({
            "user_id": pred_id,
            "name": matched_name,
            "action": action.upper(),
            "timestamp": now_str
        })
        return {"status": "success", "name": matched_name, "distance": int(dist)}

    return JSONResponse({"status": "error", "message": f"Unknown Face (dist {int(dist)})"}, status_code=401)