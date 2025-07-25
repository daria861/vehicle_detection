import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json

def main():
    # Paths
    train_dir = 'stanford-car-dataset-by-classes-folder/car_data/car_data/train'
    test_dir = 'stanford-car-dataset-by-classes-folder/car_data/car_data/test'

    print("Working directory:", os.getcwd())
    print("Train classes:", os.listdir(train_dir))
    print("Example images:", os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0])))

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # On Windows, try num_workers=0 if you still get issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Save class mapping
    with open("models/class_to_idx.json", "w") as f:
        json.dump(train_dataset.class_to_idx, f)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)

    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "models/car_make_model_efficientnet.pth")
    print("Model and class mapping saved.")

#if __name__ == "__main__":
#    main()