import torch
from torchvision import transforms, models
import cv2
from PIL import Image
import numpy as np
import json

def load_classifier(model_path, class_map_path, device='cpu'):
    """
    Load the EfficientNet classifier for car make/model recognition.
    """
    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    return model, class_names

def preprocess_crop(img, input_size=(224, 224)):
    """
    Preprocess a cropped car image for model input.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return torch.zeros(1, 3, input_size[0], input_size[1])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor

def predict_make_model(model, crop_img, class_names, device='cpu'):
    """
    Predict the car make/model given a cropped car image.
    """
    try:
        model.eval()
        input_tensor = preprocess_crop(crop_img).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred_idx = pred.item()
            if pred_idx >= len(class_names):
                return "Unknown"
            return class_names[pred_idx]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"