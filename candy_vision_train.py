import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# === Config ===
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Image Transform (Used in Training + Prediction) ===
candy_train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Used during prediction or evaluation
candy_eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def create_model(num_classes):
    """Creates and returns a pre-trained EfficientNet model."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze feature extractor layers
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model.to(DEVICE)

def load_dataset(data_dir, transform):
    """Loads dataset and returns a DataLoader and class names."""
    dataset = datasets.ImageFolder(data_dir, transform=transform)
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
    dataloader, class_names = load_dataset(data_dir, candy_train_transform)
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
    img = candy_eval_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        predicted = torch.argmax(outputs, 1).item()

    return class_names[predicted]
import random

def evaluate_model(model, dataset_dir, class_names, sample_size=100):
    """Evaluates the model on a random sample of images."""
    dataset = datasets.ImageFolder(dataset_dir, transform=candy_eval_transform)

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
def detect_candies_yolo(image_path, yolo_weights="runs/detect/train7/weights/best.pt"):
    model = YOLO(yolo_weights)
    results = model.predict(image_path, imgsz=640, conf = 0.5)[0]
    results.show()
    results.save("data/temp/board_annotated.png")
    # Each detection: xyxy, confidence, class
    detections_candy = []
    detections_gap = []
    detections_loader = []
    detections_objective = []
    for box in results.boxes:
        class_id = int(box.cls[0].item())


        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        if class_id == 0:
            detections_candy.append(xyxy)
        elif class_id == 1:
            detections_gap.append((xyxy, "gap"))
        elif class_id == 2:
            detections_loader.append((xyxy, "loader"))
        elif class_id == 3:
            detections_objective.append((xyxy, "objective"))
    print(f"Detected {len(detections_candy)} candies, {len(detections_gap)} gaps, {len(detections_loader)} loaders, {len(detections_objective)} objectives.")


    return detections_candy, detections_gap, detections_loader  # List of [x1, y1, x2, y2]
def classify_candies(image_path, detections, model, class_names):
    image = Image.open(image_path).convert("RGB")
    predictions = []

    for box in detections:
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        crop = candy_eval_transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(crop)
            pred = torch.argmax(output, 1).item()
            predictions.append((box, class_names[pred]))

    return predictions  # List of tuples: (box, class_name)
def cluster_detections_by_rows(detections, tolerance=20):
    """Groups detections into rows by Y-coordinate proximity."""
    rows = []

    for det in sorted(detections, key=lambda d: (d[0][1] + d[0][3]) / 2):  
        center_y = (det[0][1] + det[0][3]) / 2
        placed = False

        for row in rows:
            row_center = sum((box[1] + box[3]) / 2 for box, _ in row) / len(row)
            if abs(center_y - row_center) < tolerance:
                row.append(det)
                placed = True
                break

        if not placed:
            rows.append([det])

    # Sort each row left to right
    for row in rows:
        row.sort(key=lambda d: (d[0][0] + d[0][2]) / 2)

    return rows  # list of rows, where each row is a list of (box, class)
def auto_expand_dataset_from_yolo(
    image_dirs=["data/images/train", "data/images/val"],
    yolo_model_path="runs/detect/train7/weights/best.pt",
    candy_dataset_path="candy_dataset",
    max_per_class=50
):
    model, class_names = load_or_train_model(candy_dataset_path, "candy_classifier.pth")
    yolo = YOLO(yolo_model_path)

    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            continue

        for filename in os.listdir(dir_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(dir_path, filename)
            image = Image.open(image_path).convert("RGB")
            results = yolo.predict(image_path, imgsz=640, conf=0.5)[0]

            for box in results.boxes:
                class_id = int(box.cls[0].item())
                if class_id != 0:  # Only process candies
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = map(int, xyxy)
                crop = image.crop((x1, y1, x2, y2))
                transformed = candy_eval_transform(crop).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(transformed)
                    pred_idx = torch.argmax(output, 1).item()
                    class_name = class_names[pred_idx]

                target_folder = os.path.join(candy_dataset_path, class_name)
                os.makedirs(target_folder, exist_ok=True)
                if len(os.listdir(target_folder)) >= max_per_class:
                    continue  # Skip if already at max

                # Save the cropped image
                new_name = f"{filename.split('.')[0]}_{x1}_{y1}_{x2}_{y2}.png"
                save_path = os.path.join(target_folder, new_name)
                crop.save(save_path)

            print(f"Processed: {filename}")

if __name__ == "__main__":
    """
    yolo_model_path = "runs/detect/train7/weights/best.pt"
    candy_model_path = "candy_classifier.pth"
    data_dir = "candy_dataset"
    screenshot_path = "data/temp/board107.png"
    # Load classifier
    model, class_names = load_or_train_model(data_dir, candy_model_path)

    # Step 1: Detect
    candies_box, gap_box, loader_box = detect_candies_yolo(screenshot_path, yolo_model_path)

    # Step 2: Classify
    classified = classify_candies(screenshot_path, candies_box, model, class_names)
    combined = classified + gap_box  # Combine candy and gap detections
    # Step 3 (optional): Grid structure
    grid = cluster_detections_by_rows(combined)

    # Print or use the grid
    for i, row in enumerate(grid):
        print(f"Row {i + 1}: {[label for _, label in row]}")"""
    auto_expand_dataset_from_yolo(
        image_dirs=["data/images/train", "data/images/val"],
        yolo_model_path="runs/detect/train7/weights/best.pt",
        candy_dataset_path="candy_dataset",
        max_per_class=50
    )
    print("✅ Dataset expansion completed.")

