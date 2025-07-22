import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_classifier(model_path, device='cpu'):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def preprocess_crop(crop):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(crop[..., ::-1])  # BGR to RGB
    return transform(img).unsqueeze(0)

def predict_make_model(model, crop, class_names, device='cpu'):
    input_tensor = preprocess_crop(crop).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred = logits.argmax(1).item()
    return class_names[pred]