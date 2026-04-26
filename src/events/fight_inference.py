import cv2
from fight_classifier import FightDetector
from ultralytics import YOLO
detector = FightDetector("models/fight_model.pth")
from collections import deque

buffer_size = 10
threshold = 7

frame_buffer = deque(maxlen=buffer_size)

cap = cv2.VideoCapture(r"C:\Users\vpokh\Downloads\V_100.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    roi = frame[int(h*0.4):h, :]

    label, conf = detector.predict(roi)

    if label == "violence" and conf > 0.65:
         frame_buffer.append(1)
    else:
         frame_buffer.append(0)
# Count positives
    violence_count = sum(frame_buffer)

    if violence_count >= threshold:
     cv2.putText(frame, "FIGHT DETECTED!", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
     
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()