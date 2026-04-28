import os
import cv2

DATASET_PATH = r"C:\Users\vpokh\Downloads\FIRE-SMOKE-DATASET\FIRE-SMOKE-DATASET"
OUTPUT_PATH = "data/fire_yolo"

classes = {"Fire": 0, "Smoke": 1}

for split in ["train", "test"]:
    for cls in os.listdir(os.path.join(DATASET_PATH, split)):
        
        if cls == "Neutral":
            continue  # skip neutral (no object)

        cls_id = classes[cls]

        img_dir = os.path.join(DATASET_PATH, split, cls)
        out_img_dir = os.path.join(OUTPUT_PATH, "images", split)
        out_lbl_dir = os.path.join(OUTPUT_PATH, "labels", split)

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            # Copy image
            cv2.imwrite(os.path.join(out_img_dir, img_name), img)

            # Full image bbox
            x_center = 0.5
            y_center = 0.5
            width = 1.0
            height = 1.0

            label_path = os.path.join(out_lbl_dir, img_name.replace(".jpg", ".txt"))

            with open(label_path, "w") as f:
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")