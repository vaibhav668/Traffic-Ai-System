from ultralytics import YOLO
import cv2
import time

model=YOLO("yolov8n.pt")

VEHICLE_CLASSES= [2,3,5,7]

cap=cv2.VideoCapture(0)

prev_time=0

while True:
    ret,frame=cap.read()

    if not ret:
        break

    results = model.track(frame,persist=True)
    for r in results:
        if r.boxes is None:
            continue
        boxes=r.boxes

        for box in boxes:
            cls_id=int(box.cls[0])

            if cls_id in VEHICLE_CLASSES:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                track_id=int(box.id[0]) if box.id is not None else -1
                label = f"ID {track_id} | {model.names[cls_id]}"

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    