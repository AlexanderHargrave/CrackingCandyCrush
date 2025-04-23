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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Image Transform (Used in Training + Prediction) ===
candy_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def create_model(num_classes):
    """Builds a ResNet18 model for classification."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def load_dataset(data_dir):
    """Loads dataset and returns a DataLoader and class names."""
    dataset = datasets.ImageFolder(data_dir, transform=candy_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, dataset.classes

def train_model(model, dataloader, num_epochs=EPOCHS, lr=LR):
    """Trains the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader):
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

    return model

def load_or_train_model(data_dir='candy_dataset', model_path='candy_classifier.pth'):
    """Loads the model if it exists, or trains and saves it."""
    dataloader, class_names = load_dataset(data_dir)
    model = create_model(len(class_names))

    if os.path.exists(model_path):
        print("✅ Pre-trained model found. Loading...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("⚠️ No pre-trained model found. Training a new one...")
        model = train_model(model, dataloader)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

    model.eval()
    return model, class_names

def predict_image(image_path, model, class_names):
    """Predicts a single image."""
    img = Image.open(image_path).convert('RGB')
    img = candy_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        predicted = torch.argmax(outputs, 1).item()

    return class_names[predicted]
import random

def evaluate_model(model, dataset_dir, class_names, sample_size=100):
    """Evaluates the model on a random sample of images."""
    dataset = datasets.ImageFolder(dataset_dir, transform=candy_transform)

    all_indices = list(range(len(dataset)))
    sample_indices = random.sample(all_indices, min(sample_size, len(dataset)))
    correct = 0

    for idx in sample_indices:
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img)
            predicted = torch.argmax(outputs, 1).item()

        if predicted == label:
            correct += 1

    accuracy = 100 * correct / len(sample_indices)
    print(f"🔍 Accuracy on {len(sample_indices)} random samples: {accuracy:.2f}%")
    return accuracy

# === Standalone Usage Example ===
if __name__ == "__main__":
    DATASET_DIR = 'candy_dataset'
    MODEL_PATH = 'candy_classifier.pth'
    TEST_IMAGE = 'candy_dataset/frosting4/f07.png'  # Replace with any test image

    model, class_names = load_or_train_model(DATASET_DIR, MODEL_PATH)
    evaluate_model(model, DATASET_DIR, class_names)
    if os.path.exists(TEST_IMAGE):
        prediction = predict_image(TEST_IMAGE, model, class_names)
        print(f"🧠 Predicted: {prediction}")
    else:
        print("⚠️ Test image not found.")
