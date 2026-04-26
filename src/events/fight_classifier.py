import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2

class FightDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model architecture
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.class_names = checkpoint["class_names"]

        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, frame):
        image = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, pred = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0][pred].item()

            return self.class_names[pred.item()], confidence