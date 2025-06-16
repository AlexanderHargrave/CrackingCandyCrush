import os
import torch
import torch.nn as nn
import numpy as np
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
import matplotlib.pyplot as plt
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

def train_model(model_name, train_loader, val_loader, num_epochs=EPOCHS, lr=LR, patience=5, target = "candy"):
    model = get_model(model_name, len(train_loader.dataset.dataset.classes))
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    # For tracking
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

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
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
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
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # ====== EARLY STOPPING LOGIC ======
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ====== SAVE TRAINING CURVES ======
    os.makedirs("graphs", exist_ok=True)
    epochs_range = range(1, len(train_accuracies) + 1)
    if target == "candy":
        classifier = "Candy"
    elif target == "objective":
        classifier = "Objective"
    elif target == "loader":
        classifier = "Loader"
    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
    plt.plot(epochs_range, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy for {model_name} Model on {classifier} Classification")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join("graphs", f"{model_name}_accuracy_objective.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"📈 Accuracy graph saved to '{acc_path}'")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss for {model_name} Model on {classifier} Classification")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join("graphs", f"{model_name}_loss_objective.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"📉 Loss graph saved to '{loss_path}'")

    return model


def load_or_train_model(data_dir='candy_dataset', model_path='candy_classifier_efficientnet_b0.pth', 
                        num_epochs=EPOCHS, lr=LR, model_name="efficientnet_b0", target = "candy"):
    """Loads the model if it exists, or trains and saves it."""
    train_loader, val_loader, class_names = load_dataset(data_dir, candy_train_transform)
    model = get_model(model_name, len(class_names))

    if os.path.exists(model_path):
        print("✅ Pre-trained model found. Loading...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(model_name)
        print("⚠️ No pre-trained model found. Training a new one...")
        model = train_model(model_name, train_loader, val_loader, num_epochs=num_epochs, lr=lr, target=target)
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
            detections_candy.append((xyxy, "candy"))
        elif class_id == 1:
            detections_gap.append((xyxy, "gap"))
        elif class_id == 2:
            detections_loader.append((xyxy, "loader"))
        elif class_id == 3:
            detections_objective.append((xyxy, "objective"))
    print(f"Detected {len(detections_candy)} candies, {len(detections_gap)} gaps, {len(detections_loader)} loaders, {len(detections_objective)} objectives.")


    return detections_candy, detections_gap, detections_loader, detections_objective  # List of [x1, y1, x2, y2]
def classify_candies(image_path, detections, models, class_names, update = False):
    image = Image.open(image_path).convert("RGB")
    predictions = []
    max_per_class = 50  # Limit per class to avoid dataset bloat
    for box, _ in detections:
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        crop_transform = candy_eval_transform(crop).unsqueeze(0).to(DEVICE)
        model_votes = []
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(crop_transform)
                pred = torch.argmax(output, 1).item()
                model_votes.append(pred)
        modal_pred = Counter(model_votes).most_common(1)[0][0]
        predictions.append((box, class_names[modal_pred]))
        if update:
            target_folder = os.path.join("candy_dataset", class_names[modal_pred])
            os.makedirs(target_folder, exist_ok=True)
            if len(os.listdir(target_folder)) >= max_per_class:
                continue
            base_name = f"{class_names[modal_pred]}_{len(os.listdir(target_folder)) + 1}"
            new_name = f"{base_name}.png"
            count = 1
            while os.path.exists(os.path.join(target_folder, new_name)):
                new_name = f"{base_name}_{count}.png"
                count += 1

            save_path = os.path.join(target_folder, new_name)
            # print(f"Saving crop to {save_path}")
            
            crop.save(save_path)
    return predictions  # List of tuples: (box, class_name)
import pytesseract
def cluster_detections_by_rows(candy_detections, gap_detections, loader_detections, tolerance=40):
    """
    Builds a full grid using candy/gap detections, estimates missing tiles as gaps,
    and assigns loader detections to the grid based on proximity to candies below.
    """
    # Combine and normalize
    all_detections = [(box, label) for box, label in candy_detections + gap_detections]

    if not candy_detections:
        print("⚠️ No candies detected.")
        return []

    # === Step 1: Estimate board dimensions from candy positions ===
    x_coords = [box[0][0] for box in candy_detections]
    y_coords = [box[0][1] for box in candy_detections]
    x_coords += [box[0][2] for box in candy_detections]
    y_coords += [box[0][3] for box in candy_detections]
    # pring average candy size, both x and y
    #avg_candy_width = np.mean([box[0][2] - box[0][0] for box in candy_detections])
    #avg_candy_height = np.mean([box[0][3] - box[0][1] for box in candy_detections])
    avg_candy_width = 66
    avg_candy_height = 73
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    grid_width = max_x - min_x
    grid_height = max_y - min_y
    print(f"🗺️ Detected grid dimensions: {grid_width}x{grid_height} pixels")

    num_cols = round(grid_width / avg_candy_width)
    num_rows = round(grid_height / avg_candy_height)

    if num_cols == 0 or num_rows == 0:
        print("⚠️ Could not infer grid dimensions.")
        return []

    print(f"📐 Estimated Grid: {num_rows} rows x {num_cols} cols")

    # === Step 2: Build grid shape ===
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    for row in range(num_rows):
        for col in range(num_cols):
            # Estimated center of each cell
            center_x = int(min_x + (col + 0.5) * avg_candy_width)
            center_y = int(min_y + (row + 0.5) * avg_candy_height)

            # Find closest detection
            best_dist = float("inf")
            best_match = None
            best_type = None

            for (box, label) in all_detections:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dist = np.hypot(cx - center_x, cy - center_y)
                if dist < best_dist and dist < tolerance:
                    best_dist = dist
                    best_match = (box, label)
                    best_type = label

            # Prioritize candy over gap
            if best_match and best_type != "gap":
                grid[row][col] = best_match
            elif best_match:
                grid[row][col] = best_match
            else:
                # No match found — assume gap
                grid[row][col] = ([
                    int(center_x - avg_candy_width / 2),
                    int(center_y - avg_candy_height / 2),
                    int(center_x + avg_candy_width / 2),
                    int(center_y + avg_candy_height / 2)
                ], "gap")

    # === Step 3: Assign loaders to rows above matching candies ===
    for loader_box, loader_label in loader_detections:
        lx1, ly1, lx2, ly2 = loader_box
        loader_cx = (lx1 + lx2) / 2
        loader_cy = (ly1 + ly2) / 2

        best_candy_pos = None
        best_dist = float('inf')

        for row in range(num_rows):
            for col in range(num_cols):
                box, label = grid[row][col]
                if label != "gap":
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dy = cy - loader_cy
                    dx = abs(cx - loader_cx)
                    if dy > 0 and dy < 100 and dx < 50 and dy + dx < best_dist:
                        best_dist = dy + dx
                        best_candy_pos = (row, col, label)

        if best_candy_pos:
            row, col, candy_label = best_candy_pos
            loader_type = f"loader of {loader_label}"
            # Place loader above candy if possible
            if row > 0:
                grid[row - 1][col] = (loader_box, loader_type)
            # If loader above row 0, i want to generate a new row above it, where it will be placed
            elif row == 0:
                # Create a new row above
                new_row = [(loader_box, loader_type) if c == col else (None, "gap") for c in range(num_cols)]
                grid.insert(0, new_row)
                num_rows += 1
                # Shift existing rows down
                #for r in range(1, num_rows):
                    #grid[r] = grid[r - 1]

    return grid



def auto_expand_dataset_from_yolo(
    image_dirs=["data/images/train", "data/images/val"],
    yolo_model_path="runs/detect/train7/weights/best.pt",
    candy_dataset_path="candy_dataset",
    model_names=["efficientnet_b0", "efficientnet_b3", "resnet18", "resnet34", "resnet50"],
    max_per_class=50,
    num_epochs=50
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load class names ===
    _, _, class_names = load_dataset(candy_dataset_path, candy_eval_transform)
    # === Load ensemble models ===
    models_list = []
    for model_name in model_names:
        # train_loader, val_loader, _ = load_dataset(data_dir, candy_train_transform)
        model,_ = load_or_train_model(data_dir = candy_dataset_path, model_path=f"candy_classifier_{model_name}.pth", num_epochs=num_epochs, model_name=model_name)
        model.to(DEVICE)
        model.eval()
        models_list.append(model)

    # === Go through each image in provided directories ===
    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            continue

        for filename in os.listdir(dir_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(dir_path, filename)
            candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(image_path, yolo_model_path)
            classified = classify_candies(image_path, candies_box, models_list, class_names, update=True)
# This function goes through all images in data/images/train and data/images/val and data/temp and gets all the objective and loader images, saves them in objectives and loaders folders
def get_objective_loader_images(images_dir=["data/temp","data/images/train", "data/images/val"], objective_dir="objectives", loader_dir="loader"):
    os.makedirs(objective_dir, exist_ok=True)
    os.makedirs(loader_dir, exist_ok=True)
    yolo_model_path = "runs/detect/train7/weights/best.pt"
    for dir_path in images_dir:
        if not os.path.exists(dir_path):
            continue

        for filename in os.listdir(dir_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(dir_path, filename)
            candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(image_path, yolo_model_path)
            for box, label in loader_box:
                x1, y1, x2, y2 = box
                crop = Image.open(image_path).convert("RGB").crop((x1, y1, x2, y2))
                save_path = os.path.join(loader_dir, f"{label}_{filename}")
                if os.path.exists(save_path):
                    base_name, ext = os.path.splitext(save_path)
                    count = 1
                    while os.path.exists(save_path):
                        save_path = f"{base_name}_{count}{ext}"
                        count += 1
                crop.save(save_path)
            for box, label in objective_box:
                x1, y1, x2, y2 = box
                crop = Image.open(image_path).convert("RGB").crop((x1, y1, x2, y2))
                save_path = os.path.join(objective_dir, f"{label}_{filename}")
                if os.path.exists(save_path):
                    base_name, ext = os.path.splitext(save_path)
                    count = 1
                    while os.path.exists(save_path):
                        save_path = f"{base_name}_{count}{ext}"
                        count += 1
                crop.save(save_path)
            
if __name__ == "__main__":
    yolo_model_path = "runs/detect/train7/weights/best.pt"
    data_dir = "candy_dataset"
    screenshot_path = "data/images/train/board122.png"
    sample_eval_size = 1

    model_names = ["efficientnet_b0", "efficientnet_b3", "resnet18", "resnet34", "resnet50"]
    short_model_names = ["efficientnet_b0", "resnet18", "resnet34"]

    num_epochs = 50
    _, _, candy_class_names = load_dataset(data_dir, candy_eval_transform)

    # ===== CANDY CLASSIFICATION =====
    models_list = []
    for model_name in model_names:
        print(f"\n🚀 Training {model_name.upper()} for candy classification")
        model_path = f"candy_classifier_{model_name}.pth"
        model, _ = load_or_train_model(data_dir, model_path=model_path, num_epochs=num_epochs, model_name=model_name)
        model.to(DEVICE)
        #torch.save(model.state_dict(), model_path)
        model.eval()
        models_list.append(model)

        acc = evaluate_model(model, data_dir, candy_class_names, sample_size=sample_eval_size)
        print(f"🍬 Candy Accuracy: {acc:.2f}%")

    # ===== OBJECTIVE CLASSIFICATION =====
    print("\n🎯 Starting objective classification...")
    _, _, objective_class_names = load_dataset("objectives", candy_eval_transform)
    objective_models_list = []
    objective_data_dir = "objectives"
    for model_name in short_model_names:
        print(f"\n🚀 Training {model_name.upper()} for objective classification")
        model_path = f"objective_classifier_{model_name}.pth"
        model, _ = load_or_train_model(objective_data_dir, model_path=model_path, num_epochs=num_epochs,
                                       model_name=model_name, target="objective")
        model.to(DEVICE)
        #torch.save(model.state_dict(), model_path)
        model.eval()
        objective_models_list.append(model)

        acc = evaluate_model(model, objective_data_dir, objective_class_names, sample_size=sample_eval_size)
        print(f"🎯 Objective Accuracy: {acc:.2f}%")

    # ===== LOADER CLASSIFICATION =====
    print("\n📦 Starting loader classification...")
    _, _, loader_class_names = load_dataset("loader", candy_eval_transform)
    loader_models_list = []
    loader_data_dir = "loader"
    for model_name in short_model_names:
        print(f"\n🚀 Training {model_name.upper()} for loader classification")
        model_path = f"loader_classifier_{model_name}.pth"
        model, _ = load_or_train_model(loader_data_dir, model_path=model_path, num_epochs=num_epochs,
                                       model_name=model_name, target="loader")
        model.to(DEVICE)
        #torch.save(model.state_dict(), model_path)
        model.eval()
        loader_models_list.append(model)

        acc = evaluate_model(model, loader_data_dir, loader_class_names, sample_size=sample_eval_size)
        print(f"📦 Loader Accuracy: {acc:.2f}%")

    # ===== DETECTION =====
    candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(screenshot_path, yolo_model_path)

    # ===== CLASSIFICATION =====
    candy_classified = classify_candies(screenshot_path, candies_box, models_list, candy_class_names, update=False)
    gap_classified = []
    for box, _ in gap_box:
        gap_classified.append((box, "gap"))
    objective_classified = classify_candies(screenshot_path, objective_box, objective_models_list, objective_class_names, update=False)

    loader_classified = classify_candies(screenshot_path, loader_box, loader_models_list, loader_class_names, update=False)

    # Print objectives
    print("\n🎯 Objectives detected:")
    for box, label in objective_classified:
        x1, y1, x2, y2 = box
        print(f"Objective: {label} at ({x1}, {y1}, {x2}, {y2})")
    
    # ===== GRID STRUCTURING =====
    grid = cluster_detections_by_rows(candy_classified, gap_classified, loader_classified, tolerance=40)
    for i, row in enumerate(grid):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    """
    print(f"Expanding dataset iteration...")
    auto_expand_dataset_from_yolo(
        image_dirs=["data/images/train", "data/images/val"],
        yolo_model_path="runs/detect/train7/weights/best.pt",
        candy_dataset_path="candy_dataset",
        max_per_class=50
    )
    print("✅ Dataset expansion completed.")
    # run this to get all the objective and loader images
    print("Collecting objective and loader images...")
    get_objective_loader_images(images_dir=["data/temp","data/images/train", "data/images/val"], objective_dir="objectives", loader_dir="loader")
    print("✅ Objective and loader images collected.")"""

