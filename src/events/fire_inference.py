import cv2
from collections import deque
from fire_classifier import FireDetector

detector = FireDetector("models/fire_model.pth")

cap = cv2.VideoCapture(0)

# 🔧 Improved temporal settings
buffer_size = 20
threshold = 10
frame_buffer = deque(maxlen=buffer_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 🔥 Better grid (more precise)
    grid_size = 3
    patch_h = h // grid_size
    patch_w = w // grid_size

    fire_score = 0
    smoke_score = 0
    boxes = []

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w

            patch = frame[y1:y2, x1:x2]

            label, conf = detector.predict(patch)

            # 🔥 Stronger logic
            if label == "Fire" and conf > 0.75:
                fire_score += conf
                boxes.append((x1, y1, x2, y2))

            elif label == "Smoke" and conf > 0.80:
                smoke_score += conf
                boxes.append((x1, y1, x2, y2))

    # 🧠 Weighted decision
    frame_score = fire_score * 1.5 + smoke_score

    frame_buffer.append(frame_score)

    # 🚨 Final decision
    if sum(frame_buffer) > threshold:
        cv2.putText(frame, "FIRE DETECTED!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Debug info
    cv2.putText(frame, f"FireScore: {fire_score:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"SmokeScore: {smoke_score:.2f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()