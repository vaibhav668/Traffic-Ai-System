from ultralytics import YOLO

class FireYOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, frame):
        # 🔥 Increase base confidence
        results = self.model(frame, conf=0.6)

        detections = []
        h, w = frame.shape[:2]

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 📏 Compute box area ratio
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                area_ratio = box_area / frame_area

                # ✅ FILTERS (IMPORTANT)
                if conf < 0.6:
                    continue

                if area_ratio < 0.05:
                    continue  # too small → noise

                detections.append((x1, y1, x2, y2, cls_id, conf))

        return detections