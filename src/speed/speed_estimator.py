from ultralytics import YOLO
import cv2
import time
import math

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

cap = cv2.VideoCapture(0)

# Store previous positions of each ID
prev_positions = {}

prev_time=time.time()

while True:
    ret,frame=cap.read()
    if not ret:
        break
    curr_time=time.time()
    dt=curr_time-prev_time if prev_time else 0
    prev_time=curr_time

    results=model.track(frame,persist=True,verbose=False)
    
    for r in results:
        if r.boxes is None:
            continue
        
        for box in r.boxes:
            cls_id=int(box.cls[0])

            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                speed=0

                if track_id in prev_positions and dt>0:
                    px,py = prev_positions[track_id]
                    
                    dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                    speed = dist / dt 

                prev_positions[track_id] = (cx, cy)

                label = f"ID {track_id} | {model.names[cls_id]} | {int(speed)} px/s"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Speed Estimation",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()








