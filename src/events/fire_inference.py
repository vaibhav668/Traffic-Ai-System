import cv2
from collections import deque
from fire_classifier import FireDetector

detector = FireDetector("models/fire_model.pth")
cap = cv2.VideoCapture(0)

# Stronger temporal settings
buffer_size = 20
threshold = 12
frame_buffer = deque(maxlen=buffer_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Grid settings
    grid_size = 2
    patch_h = h // grid_size
    patch_w = w // grid_size

    fire_detected = 0
    boxes = []

    # 🔍 Grid-based detection
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w

            patch = frame[y1:y2, x1:x2]

            label, conf = detector.predict(patch)

            # Strong confidence filter
            if label in ["Fire", "Smoke"] and conf > 0.75:
                fire_detected += 1
                boxes.append((x1, y1, x2, y2))

    # Temporal buffer update
    frame_buffer.append(1 if fire_detected > 0 else 0)

    # 🚨 Final decision
    if sum(frame_buffer) >= threshold:
        cv2.putText(frame, "FIRE/SMOKE DETECTED!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # Draw detected boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Debug (optional but useful)
    cv2.putText(frame, f"Fire patches: {fire_detected}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()