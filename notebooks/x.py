import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO(r"C:\Users\vpokh\runs\detect\fire_smoke_detector\weights\best.pt")

cap = cv2.VideoCapture(0)

# 🔧 Temporal smoothing buffer
buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Inference with threshold
    results = model(frame, conf=0.5, iou=0.5)

    detections = results[0].boxes

    fire_detected = False

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Class names
            label = model.names[cls_id]

            # 🔥 Only consider high confidence fire/smoke
            if conf > 0.6 and label in ["fire", "smoke"]:
                fire_detected = True

    # 🧠 Temporal smoothing
    buffer.append(fire_detected)
    stable_fire = sum(buffer) > 5   # majority voting

    annotated = results[0].plot()

    # 🚨 Alert text
    if stable_fire:
        cv2.putText(annotated, "FIRE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("frame", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()