import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json

def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def main():
    # Set your dataset paths
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'

    print("Working directory:", os.getcwd())
    print("Train classes:", os.listdir(train_dir))
    print("Example images:", os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0])))

    # Strong data augmentation for training data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),               # PIL Image
        transforms.RandomHorizontalFlip(),           # PIL Image
        transforms.RandomRotation(15),               # PIL Image
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # PIL Image
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # PIL Image
        transforms.ToTensor(),                       # Converts PIL Image to tensor
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),         # TENSOR ONLY
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            # TENSOR ONLY
    ])
    # For test/val data, keep it simple
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Save class mapping
    os.makedirs("models", exist_ok=True)
    with open("models/class_to_idx.json", "w") as f:
        json.dump(train_dataset.class_to_idx, f)

    # Build EfficientNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)

    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with best checkpoint saving
    epochs = 10
    best_acc = 0.0
    best_epoch = 0
    best_model_wts = None
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
        
        # Evaluate on test set
        val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch+1
            best_model_wts = model.state_dict()  # Save best weights

    print(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    # Save the best model
    torch.save(best_model_wts, "models/car_make_model_.pth")
    print("Best model and class mapping saved.")

#if __name__ == "__main__":
#    main()