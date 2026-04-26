from ultralytics import YOLO
import cv2
import time
import math

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

cap = cv2.VideoCapture(0)

prev_positions = {}
prev_speeds = {}

prev_time = time.time()

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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    dt = curr_time - prev_time if prev_time else 0
    prev_time = curr_time

    results = model.track(frame, persist=True,verbose=False)

    current_boxes = {}
    accident_flag = False

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])

            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id.item()) if box.id is not None else -1

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                speed = 0

                if track_id in prev_positions and dt > 0:
                    px, py = prev_positions[track_id]
                    dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                    speed = dist / dt

                # Check sudden speed drop
                if track_id in prev_speeds:
                    if prev_speeds[track_id] > 50 and speed < 10:
                        accident_flag = True

                prev_positions[track_id] = (cx, cy)
                prev_speeds[track_id] = speed

                current_boxes[track_id] = [x1, y1, x2, y2]

                label = f"ID {track_id} | {int(speed)} px/s"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Check collisions using IoU
    ids = list(current_boxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            box1 = current_boxes[ids[i]]
            box2 = current_boxes[ids[j]]

            iou = compute_iou(box1, box2)

            if iou > 0.3:
                accident_flag = True

    # Display alert
    if accident_flag:
        cv2.putText(frame, "ACCIDENT DETECTED!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()