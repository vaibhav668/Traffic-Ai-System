import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2

class FireDetector:
    def __init__(self, model_path, conf_threshold=0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold

        # Load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.class_names = checkpoint["class_names"]

        self.model.to(self.device)
        self.model.eval()

        # ✅ FIXED TRANSFORM (match training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame):
        # ✅ CRITICAL FIX: OpenCV gives BGR → convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        conf = conf.item()
        pred_class = self.class_names[pred.item()]

        # ✅ CONFIDENCE FILTER (prevents false fire alerts)
        if conf < self.conf_threshold:
            return "Uncertain", conf

        return pred_class, conf