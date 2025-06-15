import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
)
from collections import Counter
from torch.utils.data import random_split
import random
# === Config ===
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.0001
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



def get_model(model_name, num_classes):
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model
def create_model(num_classes):
    """Creates and returns a pre-trained EfficientNet model."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model.to(DEVICE)

def load_dataset(data_dir, transform, val_split=0.2):
    """Splits dataset into train and validation loaders."""
    full_dataset = datasets.ImageFolder(data_dir)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, full_dataset.classes

def train_model(model, train_loader, val_loader, num_epochs=EPOCHS, lr=LR, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # ====== VALIDATION ======
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # ====== EARLY STOPPING LOGIC ======
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0  # reset patience if improved
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # Load the best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def load_or_train_model(data_dir='candy_dataset', model_path='candy_classifier.pth', 
                        num_epochs=EPOCHS, lr=LR):
    """Loads the model if it exists, or trains and saves it."""
    train_loader, val_loader, class_names = load_dataset(data_dir, candy_train_transform)
    model = create_model(len(class_names))

    if os.path.exists(model_path):
        print("✅ Pre-trained model found. Loading...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("⚠️ No pre-trained model found. Training a new one...")
        model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

    model.eval()
    return model, class_names


def evaluate_model(model, dataset_dir, class_names, sample_size=100):
    """Evaluates the model on a random sample of images."""
    dataset = datasets.ImageFolder(dataset_dir, transform=candy_eval_transform)

    all_indices = list(range(len(dataset)))
    sample_indices = random.sample(all_indices, min(sample_size, len(dataset)))
    correct = 0
    print(f"Evaluating on {len(sample_indices)} random samples...")
    model.eval()
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
def classify_candies(image_path, detections, models, class_names):
    image = Image.open(image_path).convert("RGB")
    predictions = []

    for box in detections:
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        crop = candy_eval_transform(crop).unsqueeze(0).to(DEVICE)
        model_votes = []
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(crop)
                pred = torch.argmax(output, 1).item()
                model_votes.append(pred)
        modal_pred = Counter(model_votes).most_common(1)[0][0]
        predictions.append((box, class_names[modal_pred]))

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
    model_names=["efficientnet_b0", "efficientnet_b3", "resnet18", "resnet34", "resnet50"],
    max_per_class=50
):
    # === Load class names ===
    _, _, class_names = load_dataset(candy_dataset_path, candy_eval_transform)

    # === Load ensemble models ===
    models_path = [f"candy_classifier_{name}.pth" for name in model_names]
    models = [get_model(name, len(class_names)).to(DEVICE) for name in model_names]
    for model, path in zip(models, models_path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

    # === Load YOLO model ===
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

                # === Ensemble prediction ===
                votes = []
                with torch.no_grad():
                    for model in models:
                        output = model(transformed)
                        pred_idx = torch.argmax(output, 1).item()
                        votes.append(pred_idx)

                # Majority vote
                final_class_idx = max(set(votes), key=votes.count)
                class_name = class_names[final_class_idx]

                # === Save to dataset if under limit ===
                target_folder = os.path.join(candy_dataset_path, class_name)
                os.makedirs(target_folder, exist_ok=True)
                if len(os.listdir(target_folder)) >= max_per_class:
                    continue  # Skip if already at max

                # Ensure unique filename
                base_name = f"{class_name}_{len(os.listdir(target_folder)) + 1}"
                new_name = f"{base_name}.png"
                count = 1
                while os.path.exists(os.path.join(target_folder, new_name)):
                    new_name = f"{base_name}_{count}.png"
                    count += 1

                save_path = os.path.join(target_folder, new_name)
                crop.save(save_path)

            print(f"✅ Processed: {filename}")

if __name__ == "__main__":
    yolo_model_path = "runs/detect/train7/weights/best.pt"
    data_dir = "candy_dataset"
    screenshot_path = "data/temp/board45.png"
    sample_eval_size = 1000  # Evaluation size for performance estimation

    # Hyperparameter search space
    model_names = ["efficientnet_b0","efficientnet_b3", "resnet18", "resnet34", "resnet50"]
    num_epochs = 50
    _,_, class_names = load_dataset(data_dir, candy_eval_transform)
    best_model = None
    best_config = None
    best_acc = 0
    models = []
    for model_name in model_names:
        print(f"\n🚀 Testing {model_name.upper()}")
        train_loader, val_loader, _ = load_dataset(data_dir, candy_train_transform)
        model = get_model(model_name, len(class_names))
        model,_ = load_or_train_model(data_dir, model_path=f"candy_classifier_{model_name}.pth", num_epochs=num_epochs)
        models.append(model)
        #acc = evaluate_model(model, data_dir, class_names, sample_size=sample_eval_size)
        #print(f"🔍 Accuracy: {acc:.2f}%")

        #if acc > best_acc:
            #best_acc = acc
            #best_model = model
            #best_config = model_name
        torch.save(model.state_dict(), f"candy_classifier_{model_name}.pth")
    #print(f"\n🏆 Best Model: {best_config.upper()} | Accuracy: {best_acc:.2f}%")

    # Step 1: Detect
    candies_box, gap_box, loader_box = detect_candies_yolo(screenshot_path, yolo_model_path)

    # Step 2: Classify
    classified = classify_candies(screenshot_path, candies_box, models, class_names)
    combined = classified + gap_box  # Combine candy and gap detections

    # Step 3: Optional grid structure
    grid = cluster_detections_by_rows(combined)

    for i, row in enumerate(grid):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    """
    # run multiple times to ensure dataset is expanded

    print(f"Expanding dataset iteration {_ + 1}...")
    auto_expand_dataset_from_yolo(
        image_dirs=["data/images/train", "data/images/val"],
        yolo_model_path="runs/detect/train7/weights/best.pt",
        candy_dataset_path="candy_dataset",
        max_per_class=50
    )
    print("✅ Dataset expansion completed.")"""

