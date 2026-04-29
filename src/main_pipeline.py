import cv2
import time
import math
from collections import deque
from ultralytics import YOLO
import logging
from collections import deque
conf_buffer = deque(maxlen=10)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from src.events.fire_yolo import FireYOLODetector
from src.events.fight_classifier import FightDetector

model = YOLO("yolov8n.pt")

fire_detector = YOLO(r"C:\Users\vpokh\runs\detect\fire_smoke_detector\weights\best.pt")
fight_detector = FightDetector(
    r"C:\Users\vpokh\traffic-ai-system\models\fight_model.pth"
)
VEHICLE_CLASSES = [2, 3, 5, 7]

cap = cv2.VideoCapture(r"C:\Users\vpokh\Downloads\videoplayback.mp4")


fire_buffer = deque(maxlen=20)
fight_buffer = deque(maxlen=20)
accident_buffer = deque(maxlen=10)


prev_positions = {}
prev_speeds = {}

prev_time = time.time()

import numpy as np

import numpy as np
import cv2

def is_fire_color(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 150, 150])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 150, 150])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2

    ratio = cv2.countNonZero(mask) / (patch.shape[0] * patch.shape[1])

   
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    
    return ratio > 0.25 and variance > 500

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = boxAArea + boxBArea - interArea

    return interArea / union if union > 0 else 0


def process_frame(frame):

    alerts = {
        "fire": False,
        "accident": False,
        "fight": False,
        "vehicle_count": 0
    }

    global prev_time, prev_positions, prev_speeds
    global accident_buffer, fight_buffer

    vehicle_count = 0
    fire_flag = False

    curr_time = time.time()
    dt = curr_time - prev_time if prev_time else 0
    prev_time = curr_time

  
    results = model.track(frame, persist=True, verbose=False)

    current_boxes = {}
    accident_flag = False

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])

            if cls_id in VEHICLE_CLASSES:
                vehicle_count += 1   

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id.item()) if box.id is not None else -1

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                speed = 0

                if track_id in prev_positions and dt > 0:
                    px, py = prev_positions[track_id]
                    dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                    speed = dist / dt

                if track_id in prev_speeds:
                    prev_speed = prev_speeds[track_id]

                    if prev_speed > 30:
                        drop_ratio = (prev_speed - speed) / (prev_speed + 1e-5)

                        if drop_ratio > 0.6 and speed < 10:
                            accident_flag = True

                prev_positions[track_id] = (cx, cy)
                prev_speeds[track_id] = speed

                current_boxes[track_id] = [x1, y1, x2, y2]

                label = f"ID {track_id} | {int(speed)} px/s"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    ids = list(current_boxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            iou = compute_iou(current_boxes[ids[i]], current_boxes[ids[j]])
            if iou > 0.5:
                if prev_speeds.get(ids[i], 0) > 25 and prev_speeds.get(ids[j], 0) > 25:
                    accident_flag = True

    accident_buffer.append(1 if accident_flag else 0)

    if sum(accident_buffer) >= 7:
        cv2.putText(frame, "ACCIDENT DETECTED!", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    

    results = fire_detector(frame, conf=0.5, iou=0.5)

    detections = results[0].boxes

    fire_detected = False

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            
            label = fire_detector.names[cls_id]

            
            if conf > 0.6 and label in ["fire", "smoke"]:
                fire_detected = True


    fire_buffer.append(fire_detected)
    stable_fire = sum(fire_buffer) > 5  

    annotated = results[0].plot()

   
    if stable_fire:
        cv2.putText(annotated, "FIRE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    fire_flag = sum(fire_buffer) >= 6  

    # ================= FIGHT DETECTION =================
    fight_label, fight_conf = fight_detector.predict(frame)

    if fight_label == "violence" and fight_conf > 0.85:
        fight_buffer.append(1)
    else:
        fight_buffer.append(0)

    if sum(fight_buffer) >= 12:
        cv2.putText(frame, "FIGHT DETECTED!", (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ================= ALERTS =================
    alerts["vehicle_count"] = vehicle_count
    alerts["fire"] = fire_flag
    alerts["accident"] = sum(accident_buffer) >= 7
    alerts["fight"] = sum(fight_buffer) >= 12

    return frame, alerts