import cv2
from fight_classifier import FightDetector
from ultralytics import YOLO
detector = FightDetector("models/fight_model.pth")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.predict(frame)

    if result == "violence":
        cv2.putText(frame, "FIGHT DETECTED!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    cv2.imshow("Fight Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()