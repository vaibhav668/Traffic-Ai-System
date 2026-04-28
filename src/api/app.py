from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse,Response
import cv2
import threading
import time
import numpy as np
app = FastAPI()

# ================= GLOBAL STATE =================
cap = None
current_source = None

latest_frame = None


latest_alerts = {
    "fire": False,
    "accident": False,
    "fight": False,
    "vehicle_count": 0
}


lock = threading.Lock()

# ================= IMPORT YOUR PIPELINE =================
from src.main_pipeline import process_frame

# ================= VIDEO THREAD =================
def video_loop():
    global cap, latest_frame, latest_alerts

    while True:
        if cap is None:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        processed_frame, alerts = process_frame(frame)

        with lock:
            latest_frame = processed_frame
            latest_alerts = alerts


threading.Thread(target=video_loop, daemon=True).start()

# ================= STREAM =================
def generate_frames():
    global latest_frame

    while True:
        with lock:
            frame = latest_frame

        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

            time.sleep(0.05)
            continue

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

        time.sleep(0.01)
        
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ================= ALERTS =================
@app.get("/alerts")
def get_alerts():
    with lock:
        return latest_alerts

# ================= SET WEBCAM =================
@app.post("/set_webcam")
def set_webcam():
    global cap, current_source

    if cap:
        cap.release()

    cap = cv2.VideoCapture(0)
    current_source = "webcam"

    return {"status": "webcam started"}

@app.get("/frame")
def get_frame():
    global latest_frame

    with lock:
        frame = latest_frame

    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    success, buffer = cv2.imencode('.jpg', frame)

    if not success:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
# ================= UPLOAD VIDEO =================
@app.post("/upload_video")
def upload_video(file: UploadFile = File(...)):
    global cap, current_source

    filepath = f"temp_{file.filename}"

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    if cap:
        cap.release()

    cap = cv2.VideoCapture(filepath)
    current_source = "video"

    return {"status": "video uploaded and started"}
# ================= STOP =================
@app.post("/stop")
def stop():
    global cap, current_source

    if cap:
        cap.release()
        cap = None

    current_source = None

    return {"status": "stopped"}