import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# === Config ===
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DATASET_DIR = 'candy_dataset'
MODEL_PATH = 'candy_classifier.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Load Dataset (for class names + optional training) ===
dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes
print(f"Classes: {class_names} ({len(class_names)} total)")

# === Build Model ===
def create_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

model = create_model(len(class_names))

# === Train If Needed ===
if not os.path.exists(MODEL_PATH):
    print("⚠️ No pre-trained model found. Training a new one...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
else:
    print("✅ Pre-trained model found. Loading...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

# === Prediction Function ===
def predict_image(img_path, model, class_names):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        predicted = torch.argmax(outputs, 1).item()
    return class_names[predicted]

# === Example Usage ===
if __name__ == "__main__":
    test_image = "candy_dataset/frosting4/f07.png"  # Replace with your test image path
    if os.path.exists(test_image):
        prediction = predict_image(test_image, model, class_names)
        print(f"🧠 Predicted: {prediction}")
    else:
        print("⚠️ Test image not found.")
