import os
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import pandas as pd
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
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import easyocr
from candy_simulation import find_possible_moves, extract_jelly_grid, find_all_matches, apply_move, clear_matches, update_board, merge_jelly_to_grid, ObjectivesTracker, infer_hidden_jelly_layers
from optimal_move_selection import depth_based_simulation, monte_carlo_best_move, simulate_to_completion, heuristics_softmax_best_move, expectimax, simulate_to_completion_with_ensemble, ensemble_strategy
from hybrid_mcts import hybrid_mcts
reader = easyocr.Reader(['en'], gpu=False) 
# Config
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transform 
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


# Model selection for candy classification
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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
    acc_path = os.path.join("graphs", f"{model_name}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy graph saved to '{acc_path}'")

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
    loss_path = os.path.join("graphs", f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss graph saved to '{loss_path}'")

    return model


def load_or_train_model(data_dir='candy_dataset', model_path='candy_classifier_efficientnet_b0.pth', 
                        num_epochs=EPOCHS, lr=LR, model_name="efficientnet_b0", target = "candy"):
    """Loads the model if it exists, or trains and saves it."""
    train_loader, val_loader, class_names = load_dataset(data_dir, candy_train_transform)
    model = get_model(model_name, len(class_names))

    if os.path.exists(model_path):
        print("Pre-trained model found. Loading.")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(model_name)
        print("No pre-trained model found. Training a new one.")
        model = train_model(model_name, train_loader, val_loader, num_epochs=num_epochs, lr=lr, target=target)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    model.eval()
    return model, class_names


def evaluate_model(model, dataset_dir, class_names, sample_size=100):
    """Evaluates the model or given sample size from dataset."""
    dataset = datasets.ImageFolder(dataset_dir, transform=candy_eval_transform)

    all_indices = list(range(len(dataset)))
    sample_indices = random.sample(all_indices, min(sample_size, len(dataset)))
    correct = 0
    print(f"Evaluating on {len(sample_indices)} random samples.")
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
    print(f"Accuracy on {len(sample_indices)} random samples: {accuracy:.2f}%")
    return accuracy
def detect_candies_yolo(image_path, yolo_weights="runs/detect/train7/weights/best.pt"):
    model = YOLO(yolo_weights)
    results = model.predict(image_path, imgsz=640, conf = 0.5)[0]
    #results.show()
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


    return detections_candy, detections_gap, detections_loader, detections_objective  # Lists: [x1, y1, x2, y2]
def extract_jelly_colour_range(folder):
    pixels = []
    for fname in os.listdir(folder):
        if fname.endswith(".png"):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            arr = np.array(img).reshape(-1, 3)
            pixels.append(arr)
    all_pixels = np.vstack(pixels)
    # Get mean values and the range to be detected is like 20 pixels around the mean
    mean_vals = np.mean(all_pixels, axis=0)
    min_vals = np.maximum(mean_vals - 10, 0) 
    max_vals = np.minimum(mean_vals + 10, 255)  
    return min_vals, max_vals
def extract_unique_colors(folder):
    """Extract RGB tuples from image"""
    pixels = []
    for fname in os.listdir(folder):
        if fname.endswith(".png"):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            pixels.append(np.array(img).reshape(-1, 3))
    all_pixels = np.vstack(pixels)
    unique_colors = np.unique(all_pixels, axis=0)
    reference_colors = [tuple(color) for color in unique_colors if np.all(color >= 0) and np.all(color <= 255)]
    return reference_colors
def count_matching_colors_from_patch(crop_pil, reference_colors, tolerance=3):
    crop = np.array(crop_pil).reshape(-1, 3)
    count = 0
    for ref_color in reference_colors:
        mask = np.all(np.abs(crop - ref_color) <= tolerance, axis=1)
        count += np.sum(mask)
    return count
def count_matching_jelly_pixels(crop_pil, color_range):
    crop = np.array(crop_pil).reshape(-1, 3)
    min_rgb, max_rgb = color_range
    match = np.all((crop >= min_rgb) & (crop <= max_rgb), axis=1)
    return np.sum(match)
def detect_jelly_layer(crop_pil, range1, range2,  range3, range4, thresh1=30, thresh2=30, thresh3 = 50, thresh4 = 50, candy_type = None):
    # Really dumb function to get jelly layer depending on color of pixels trained on same pixel range to identify stuff correctly. It works most of the time but not fully checked and locks mess it up.
    count1 = count_matching_colors_from_patch(crop_pil, range1)
    count2 = count_matching_colors_from_patch(crop_pil, range2)
    count3 = count_matching_colors_from_patch(crop_pil, range3)
    count4 = count_matching_colors_from_patch(crop_pil, range4, tolerance = 5)
    if "orange" in candy_type:
        thresh3 = 70
    if "lock" in candy_type:
        thresh1 = 15
        thresh2 = 15
        thresh4 = 30
    if "bubblegum" in candy_type:
        thresh4 = 10
    if "frosting" in candy_type:
        thresh1 = 20
        thresh2 = 20
        thresh3 = 20
        thresh4 = 30

    if count1 > thresh1 and count1 > count2 and count3 < count1:
        if count2 > 1 and abs(count1 - count2) < 10:
            crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
            count1 = count_matching_colors_from_patch(crop_pil, range1)
            count2 = count_matching_colors_from_patch(crop_pil, range2)
            if count1 > count2:
                return "one layer jelly"
            else:
                return "two layer jelly"
        elif count4 >= 3000:
            crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
            count1 = count_matching_colors_from_patch(crop_pil, range1)
            count2 = count_matching_colors_from_patch(crop_pil, range2)
            count3 = count_matching_colors_from_patch(crop_pil, range3)
            if count1 > thresh1 and count1 > count2 and count1 > count3:
                return "one layer jelly"
            return "no jelly"
        return "one layer jelly"
    elif count1 > thresh1 and (count1== count2 or count1 == count3):
        crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
        count1 = count_matching_colors_from_patch(crop_pil, range1)
        count2 = count_matching_colors_from_patch(crop_pil, range2)
        count3 = count_matching_colors_from_patch(crop_pil, range3)
        if count1 > thresh1 and count1 > count2 and count1 > count3:
            return "one layer jelly"
        elif count2 > thresh2 and count2 > count1 and count2 > count3:
            return "two layer jelly"
        elif count3 > thresh3 and count3 > count2 and count3 > count1:
            return "marmalade"
        else:
            return "no jelly"
                               
    elif count2 > thresh2 and count1 < count2 and count3 < count2:
        if count1 > 1 and abs(count2 - count1) < 10:
            crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
            count1 = count_matching_colors_from_patch(crop_pil, range1)
            count2 = count_matching_colors_from_patch(crop_pil, range2)
            if count2 > count1:
                return "two layer jelly"
            else:
                return "one layer jelly"
        elif count4 >= 3000:
            crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
            count1 = count_matching_colors_from_patch(crop_pil, range1)
            count2 = count_matching_colors_from_patch(crop_pil, range2)
            count3 = count_matching_colors_from_patch(crop_pil, range3)
            if count2 > thresh1 and count2 > count1 and count2 > count3:
                return "two layer jelly"
            return "no jelly"
        return "two layer jelly"
    elif count2> thresh2 and (count1== count2 or count2 == count3):
        crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
        count1 = count_matching_colors_from_patch(crop_pil, range1)
        count2 = count_matching_colors_from_patch(crop_pil, range2)
        count3 = count_matching_colors_from_patch(crop_pil, range3)
        if count1 > thresh1 and count1 > count2 and count1 > count3:
            return "one layer jelly"
        elif count2 > thresh2 and count2 > count1 and count2 > count3:
            return "two layer jelly"
        elif count3 > thresh3 and count3 > count2 and count3 > count1:
            return "marmalade"
        else:
            return "no jelly"
    elif count3 > thresh3 and count1 < count3 and count2 < count3:
        if count3 // 2 > count1 and count3//2 > count2:
            return "marmalade"
        elif count1 > count3//2:
            crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
            count1 = count_matching_colors_from_patch(crop_pil, range1)
            count2 = count_matching_colors_from_patch(crop_pil, range2)
            count3 = count_matching_colors_from_patch(crop_pil, range3)
            if count3 > count1 and count4 > count2:
                return "marmalade"
            elif count1 > count3:
                return "one layer jelly"
            elif count2 > count3:
                return "two layer jelly"
        #print(count1, count2, count3, count4, candy_type)
        elif count4 >= 3000:
            crop_pil = crop_pil.crop((1, 1, crop_pil.width-1, crop_pil.height-1))
            count3 = count_matching_colors_from_patch(crop_pil, range3)
            if count3 > thresh3:
                return "marmalade"
            return "no jelly"
        return "marmalade"
    else:
        if count1 > count4 //2 and count1 > count2  and count1 > count3:
            return "one layer jelly"   
        elif count2 > count4 //2 and count2 > count3 and count2 > count1:
            return "two layer jelly"
        else:
            if count1 > 10 and count4 > 30:
                crop_pil = crop_pil.crop((3, 3, crop_pil.width-3, crop_pil.height-3))
                count1 = count_matching_colors_from_patch(crop_pil, range1)
                count4 = count_matching_colors_from_patch(crop_pil, range4, tolerance = 5)
                if count1 > count4//2:
                    return "one layer jelly"
                else:
                    return "no jelly"
            if count2 > 10 and count4 > 30:
                crop_pil = crop_pil.crop((3, 3, crop_pil.width-3, crop_pil.height-3))
                count2 = count_matching_colors_from_patch(crop_pil, range2)
                count4 = count_matching_colors_from_patch(crop_pil, range4, tolerance = 5)
                if count2 > count4//2:
                    return "two layer jelly"
                else:
                    return "no jelly"
            if count4 <= thresh4 and count3 <= 30:
                if count1 > count2 and count1 > count3:
                    return "one layer jelly"
                elif count2 > count1 and count2 > count3:
                    return "two layer jelly"
                else:
                    if count1 == 0 and count2 == 0:
                        return "no jelly"
                    
                    count1 = count_matching_colors_from_patch(crop_pil, range1, tolerance = 5)
                    count2 = count_matching_colors_from_patch(crop_pil, range2, tolerance = 5)
                    if count1 > count2:
                        return "one layer jelly"
                    elif count2 > count1:
                        return "two layer jelly"
            return "no jelly"
def classify_candies(image_path, detections, models, class_names, update = False, check_candy = True, range1 = None, range2 = None, range3 = None, range4 = None):
    image = Image.open(image_path).convert("RGB")
    predictions = []
    max_per_class = 50 
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
        result = class_names[modal_pred]
        if check_candy:
            jelly_levels = detect_jelly_layer(crop, range1, range2, range3, range4, candy_type=result)
            if "bubblegum" in result:
                if jelly_levels == "marmalade":
                    result += "_marmalade"
            else:
                if jelly_levels == "one layer jelly":
                    result += "_jelly1"
                elif jelly_levels == "two layer jelly":
                    result += "_jelly2"
                elif jelly_levels == "marmalade":
                    if "marmalade" not in result:
                        result += "_marmalade"
                elif jelly_levels == "no jelly":
                    pass

        predictions.append((box, result))
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
    
    return predictions 
import pytesseract
def cluster_detections_by_rows(candy_detections, gap_detections, loader_detections, tolerance=40):
    """
    Builds a full grid using candy/gap detections, estimates missing tiles as gaps, and completes the rest.
    """
    all_detections = [(box, label) for box, label in candy_detections + gap_detections]

    if not candy_detections:
        print("No candies detected.")
        return []
    x_coords = [box[0][0] for box in candy_detections]
    y_coords = [box[0][1] for box in candy_detections]
    x_coords += [box[0][2] for box in candy_detections]
    y_coords += [box[0][3] for box in candy_detections]

    avg_candy_width = 66
    avg_candy_height = 73
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    num_cols = round(grid_width / avg_candy_width)
    num_rows = round(grid_height / avg_candy_height)

    if num_cols == 0 or num_rows == 0:
        print("Could not infer grid dimensions.")
        return []

    print(f"Estimated Grid: {num_rows} rows x {num_cols} cols")

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
                # No match found — assume gap and fills it in
                grid[row][col] = ([
                    int(center_x - avg_candy_width / 2),
                    int(center_y - avg_candy_height / 2),
                    int(center_x + avg_candy_width / 2),
                    int(center_y + avg_candy_height / 2)
                ], "gap")

    # Loader detecter
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
            # Place loader above candy and if top row, will get new row
            if row > 0:
                grid[row - 1][col] = (loader_box, loader_type)
            elif row == 0:
                new_row = []
                for c in range(num_cols):
                    if c == col:
                        new_row.append((loader_box, loader_type))
                    else:
                        center_x = int(min_x + (c + 0.5) * avg_candy_width)
                        center_y = int(min_y - 0.5 * avg_candy_height) 
                        gap_box = [
                            int(center_x - avg_candy_width / 2),
                            int(center_y - avg_candy_height / 2),
                            int(center_x + avg_candy_width / 2),
                            int(center_y + avg_candy_height / 2)
                        ]
                        new_row.append((gap_box, "gap"))
                grid.insert(0, new_row)
                num_rows += 1
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


    _, _, class_names = load_dataset(candy_dataset_path, candy_eval_transform)
    models_list = []
    for model_name in model_names:
        # train_loader, val_loader, _ = load_dataset(data_dir, candy_train_transform)
        model,_ = load_or_train_model(data_dir = candy_dataset_path, model_path=f"candy_classifier_{model_name}.pth", num_epochs=num_epochs, model_name=model_name)
        model.to(DEVICE)
        model.eval()
        models_list.append(model)

    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            continue
        for filename in os.listdir(dir_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(dir_path, filename)
            candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(image_path, yolo_model_path)
            classified = classify_candies(image_path, candies_box, models_list, class_names, update=True)

def get_objective_loader_images(images_dir=["data/temp","data/images/train", "data/images/val"], objective_dir="objectives", loader_dir="loader", glass_dir="glass_layers"):
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
            """
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
            """
            for box, label in candies_box:
                x1, y1, x2, y2 = box
                crop = Image.open(image_path).convert("RGB").crop((x1, y1, x2, y2))
                save_path = os.path.join(glass_dir, f"{label}_{filename}")
                if os.path.exists(save_path):
                    base_name, ext = os.path.splitext(save_path)
                    count = 1
                    while os.path.exists(save_path):
                        save_path = f"{base_name}_{count}{ext}"
                        count += 1
                crop.save(save_path)

def detect_moves(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    region = image_np[75:150, image.width//10-20:image.width//10+100]
    resized = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    result = reader.readtext(thresh, detail=0, paragraph=False, allowlist='0123456789')
    if not result:
        # Attempt with pytesseract if EasyOCR fails
        result = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789').split()
        if not result:
            return 0
    return result[0] if result else 0 


    
def get_objective_numbers(image_path, objective_detections):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    objective_numbers = []
    
    for box, label in objective_detections:
        x1, y1, x2, y2 = box

        if len(objective_detections)<3:
            region = image_np[y1:y2, x2-2:x2+50]
        else:
            # region is below instead of to the right
            region = image_np[y2-2:y2+50, x1:x2]
        resized = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        result = reader.readtext(thresh, detail=0, paragraph=False, allowlist='0123456789')
        if not result:
            result = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789').split()
            if not result:
                result = ['0']

        detected_number = None
        for text in result:
            cleaned = text.replace('&', '8').strip()
            digits_only = ''.join(filter(str.isdigit, cleaned))
            if digits_only:
                detected_number = digits_only
                break  

        if detected_number:
            objective_numbers.append((box, detected_number))

    return objective_numbers
def load_models_for_task(task_name, data_dir, model_names, num_epochs, target=None, sample_eval_size=1, eval = True):
    print(f"\nLoading models for {task_name} classification.")
    _, _, class_names = load_dataset(data_dir, candy_eval_transform)
    models_list = []

    for model_name in model_names:
        print(f"\nTraining {model_name.upper()} for {task_name} classification")
        model_path = f"{task_name}_classifier_{model_name}.pth"
        model, _ = load_or_train_model(data_dir, model_path=model_path, num_epochs=num_epochs,
                                       model_name=model_name, target=target)
        model.to(DEVICE)
        # torch.save(model.state_dict(), model_path)
        model.eval()
        models_list.append(model)
        if eval == True:
            acc = evaluate_model(model, data_dir, class_names, sample_size=sample_eval_size)
            print(f"{task_name.title()} Accuracy: {acc:.2f}%")

    return models_list, class_names
def run_move_selection_experiment(skip_existing_results=True):
    output_csv_path = "simulation_results.csv"
    graph_dir = "graphs/move_selection"
    os.makedirs(graph_dir, exist_ok=True)

    if skip_existing_results and os.path.exists(output_csv_path):
        print(f"Skipping simulation (existing results found at {output_csv_path})")
        df = pd.read_csv(output_csv_path)
    else:
        print("Running full simulation for move strategies.")

        strategies = {
            "depth": lambda *args, **kwargs: simulate_to_completion(*args, strategy_fn=depth_based_simulation, depth=2, **kwargs),
            "monte_carlo": lambda *args, **kwargs: simulate_to_completion(*args, strategy_fn=monte_carlo_best_move, simulations_per_move=3, **kwargs),
            "mcts": lambda *args, **kwargs: simulate_to_completion(*args, strategy_fn=hybrid_mcts, max_depth=2, simulations_per_move=3, **kwargs),
            "expectimax": lambda *args, **kwargs: simulate_to_completion(*args, strategy_fn=expectimax, **kwargs),
            "softmax": lambda *args, **kwargs: simulate_to_completion(*args, strategy_fn=heuristics_softmax_best_move, **kwargs)
        }

        screenshot_names = ["test1", "test2", "test3", "test5", "test6", "test7", "test8", "test9", "test11", "test12", "test14", "test15", "test16", "test24", "test26"]
        num_runs = 20
        results = []

        yolo_model_path = "runs/detect/train7/weights/best.pt"
        data_dir = "candy_dataset"
        sample_eval_size = 1
        num_epochs = 50

        model_names = ["efficientnet_b0", "efficientnet_b3", "resnet18", "resnet34", "resnet50"]
        short_model_names = ["efficientnet_b0", "resnet18", "resnet34"]

        candy_models, candy_class_names = load_models_for_task("candy", data_dir, model_names, num_epochs, "candy", sample_eval_size)
        objective_models, objective_class_names = load_models_for_task("objective", "objectives", short_model_names, num_epochs, "objective", sample_eval_size)
        loader_models, loader_class_names = load_models_for_task("loader", "loader", short_model_names, num_epochs, "loader", sample_eval_size)

        range1 = extract_unique_colors("jelly_levels/one_jelly")
        range2 = extract_unique_colors("jelly_levels/two_jelly")
        range3 = extract_unique_colors("jelly_levels/marmalade")
        range4 = extract_unique_colors("jelly_levels/zero_jelly")

        for screenshot_name in tqdm(screenshot_names, desc="Simulating Screenshots"):
            screenshot_path = f"data/test/images/{screenshot_name}.png"
            candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(screenshot_path, yolo_model_path)

            candy_classified = classify_candies(screenshot_path, candies_box, candy_models, candy_class_names, update=False, range1=range1, range2=range2, range3=range3, range4=range4)
            gap_classified = [(box, "gap") for box, _ in gap_box]
            objective_classified = classify_candies(screenshot_path, objective_box, objective_models, objective_class_names, update=False, check_candy=False)
            loader_classified = classify_candies(screenshot_path, loader_box, loader_models, loader_class_names, update=False, check_candy=False)
            objective_numbers = get_objective_numbers(screenshot_path, objective_classified)

            moves_left = int(detect_moves(screenshot_path))
            grid = cluster_detections_by_rows(candy_classified, gap_classified, loader_classified)
            candy_grid, jelly_grid = extract_jelly_grid(grid)

            objective_targets = {label: int(number) for (_, label), (_, number) in zip(objective_classified, objective_numbers)}
            jelly_grid = infer_hidden_jelly_layers(candy_grid, jelly_grid, objective_targets)

            for run_index in range(1, num_runs + 1):
                strategy_first_moves = {}
                for strategy_name, strategy_fn in strategies.items():
                    try:
                        steps_taken, summary, completed, _, _, _ = strategy_fn(
                            candy_grid=candy_grid,
                            jelly_grid=jelly_grid,
                            objective_targets=objective_targets,
                            max_steps=moves_left
                        )
                        results.append({
                            "image": screenshot_name,
                            "run": run_index,
                            "strategy": strategy_name,
                            "moves_taken": steps_taken,
                            "completed": completed
                        })
                    except Exception as e:
                        results.append({
                            "image": screenshot_name,
                            "run": run_index,
                            "strategy": strategy_name,
                            "moves_taken": -1,
                            "completed": False,
                            "error": str(e)
                        })
                        strategy_first_moves[strategy_name] = None

                try:
                    steps_taken, tracker_summary, completed, _, _, _, strategy_agreements, total_agreeable_steps = simulate_to_completion_with_ensemble(
                        candy_grid=candy_grid,
                        jelly_grid=jelly_grid,
                        objective_targets=objective_targets,
                        strategies={
                            "depth": depth_based_simulation,
                            "monte_carlo": monte_carlo_best_move,
                            "mcts": hybrid_mcts,
                            "expectimax": expectimax,
                            "softmax": heuristics_softmax_best_move
                        },
                        max_steps=moves_left
                    )
                    results.append({
                        "image": screenshot_name,
                        "run": run_index,
                        "strategy": "ensemble",
                        "moves_taken": steps_taken,
                        "completed": completed,
                        "first_move": None
                    })
                    for strat_name in strategies.keys():
                        agree_count = strategy_agreements.get(strat_name, 0)
                        results.append({
                            "image": screenshot_name,
                            "run": run_index,
                            "strategy": strat_name,
                            "ensemble_agreement_count": agree_count,
                            "ensemble_total_considered": total_agreeable_steps,
                            "agreement_rate": agree_count / total_agreeable_steps if total_agreeable_steps > 0 else None
                        })
                except Exception as e:
                    results.append({
                        "image": screenshot_name,
                        "run": run_index,
                        "strategy": "ensemble",
                        "moves_taken": -1,
                        "completed": False,
                        "error": str(e)
                    })

        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"\nSaved simulation results to {output_csv_path}")

    agreement_summary = (
        df[df["agreement_rate"].notnull()]
        .groupby("strategy")
        .agg(
            total_agreements=("ensemble_agreement_count", "sum"),
            total_considered=("ensemble_total_considered", "sum")
        )
        .reset_index()
    )
    df_filtered = df[df["moves_taken"].notnull() & df["completed"].notnull()]

    # Pivot to compare each strategy with ensemble per image/run
    comparison_records = []

    grouped = df_filtered.groupby(["image", "run"])

    for (image, run), group in grouped:
        ensemble_row = group[group["strategy"] == "ensemble"]
        if ensemble_row.empty:
            continue
        ensemble_completed = ensemble_row["completed"].values[0]
        ensemble_moves = ensemble_row["moves_taken"].values[0]

        for _, row in group.iterrows():
            strategy = row["strategy"]
            if strategy == "ensemble":
                continue

            completed = row["completed"]
            moves = row["moves_taken"]

            if completed and not ensemble_completed:
                result = "Outperformed"
            elif completed and ensemble_completed:
                if moves < ensemble_moves:
                    result = "Outperformed"
                elif moves == ensemble_moves:
                    result = "Tie"
                else:
                    result = "Worse"
            else:
                result = "Worse"

            comparison_records.append({
                "image": image,
                "run": run,
                "strategy": strategy,
                "completed": completed,
                "moves_taken": moves,
                "ensemble_completed": ensemble_completed,
                "ensemble_moves": ensemble_moves,
                "result": result
            })

    comparison_df = pd.DataFrame(comparison_records)

    summary = comparison_df.groupby("strategy")["result"].value_counts().unstack(fill_value=0).reset_index()

    result_cols = [col for col in summary.columns if col not in ['strategy']]

    summary["Outperformance Rate (%)"] = (
        100 * summary.get("Outperformed", 0) / summary[result_cols].sum(axis=1)
    ).round(2)
    print(summary)
    
    agreement_summary["agreement_rate"] = (agreement_summary["total_agreements"] / agreement_summary["total_considered"]) * 100
    df = df[df["moves_taken"] >= 0]
    completed_df = df[df["completed"] == True]

    avg_moves_summary = completed_df.groupby("strategy").agg(
        avg_moves=("moves_taken", "mean"),
        total_moves=("moves_taken", "sum"),
    ).reset_index()

    completion_summary = df.groupby("strategy").agg(
        completion_rate=("completed", "mean"),
        runs=("completed", "count"),
        completions=("completed", "sum")
    ).reset_index()

    summary = pd.merge(avg_moves_summary, completion_summary, on="strategy", how="outer")
    summary["completion_rate"] = (summary["completion_rate"] * 100).round(2)

    print("\nOverall Strategy Comparison")
    print(summary.sort_values(by="completion_rate", ascending=False))
    print("\nEnsemble Agreement Rate Summary")
    if agreement_summary.empty:
        print("No agreement rate data found.")
    else:
        print(agreement_summary.sort_values("agreement_rate", ascending=False))

    # PLOTS
    move_plot_path = os.path.join(graph_dir, "avg_moves_per_strategy.png")
    completion_plot_path = os.path.join(graph_dir, "completion_rate_per_strategy.png")
    agreement_plot_path = os.path.join(graph_dir, "ensemble_agreement_rate.png")

    plot_bar(
        summary, x="strategy", y="avg_moves",
        title="Average Moves Taken per Strategy (Completed Runs)",
        ylabel="Average Moves",
        output_path=move_plot_path,
        y_lim_pad=0.05
    )

    # 2. Completion Rate
    plot_bar(
        summary, x="strategy", y="completion_rate",
        title="Completion Rate per Strategy (%)",
        ylabel="Completion Rate (%)",
        output_path=completion_plot_path,
        y_lim_pad=0.05
    )

    # 3. Agreement Rate
    plot_bar(
        agreement_summary, x="strategy", y="agreement_rate",
        title="Agreement with Ensemble Decision (%)",
        ylabel="Agreement Rate (%)",
        output_path=agreement_plot_path,
        y_lim_pad=0.05
    )

sns.set_theme(style="whitegrid")
def plot_bar(df, x, y, title, ylabel, output_path, y_lim_pad=0.05):
    plt.figure(figsize=(10, 6))
    sorted_df = df.sort_values(y, ascending=False)

    # Dynamically create palette based on the number of unique categories
    n_colors = sorted_df[x].nunique()
    palette = sns.color_palette("Set2", n_colors)

    ax = sns.barplot(
        data=sorted_df,
        x=x,
        y=y,
        hue=x,             # Required to apply palette safely
        palette=palette,
        legend=False       # No redundant legend since x-axis already labels
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.title(title, fontsize=14, weight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=15)

    y_min = sorted_df[y].min()
    y_max = sorted_df[y].max()
    if y_min > 0 and y_max / y_min < 1.2:
        plt.ylim(y_min - y_lim_pad * y_max, min(100, y_max + y_lim_pad * y_max))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")
if __name__ == "__main__":
    """
    yolo_model_path = "runs/detect/train7/weights/best.pt"
    data_dir = "candy_dataset"
    screenshot_path = "data/test/images/test24.png"
    sample_eval_size = 1

    model_names = ["efficientnet_b0", "efficientnet_b3", "resnet18", "resnet34", "resnet50"]
    short_model_names = ["efficientnet_b0", "resnet18", "resnet34"]

    num_epochs = 50

    # CANDY CLASSIFICATION 
    candy_models, candy_class_names = load_models_for_task(
        task_name="candy",
        data_dir="candy_dataset",
        model_names=model_names,
        num_epochs=num_epochs,
        target="candy",
        sample_eval_size=sample_eval_size
    )

    objective_models, objective_class_names = load_models_for_task(
        task_name="objective",
        data_dir="objectives",
        model_names=short_model_names,
        num_epochs=num_epochs,
        target="objective",
        sample_eval_size=sample_eval_size
    )
    loader_models, loader_class_names = load_models_for_task(
        task_name="loader",
        data_dir="loader",
        model_names=short_model_names,
        num_epochs=num_epochs,
        target="loader",
        sample_eval_size=sample_eval_size
    )

    # detectiom
    candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(screenshot_path, yolo_model_path)
    range1 = extract_unique_colors("jelly_levels/one_jelly")
    range2 = extract_unique_colors("jelly_levels/two_jelly")
    range3 = extract_unique_colors("jelly_levels/marmalade")
    range4 = extract_unique_colors("jelly_levels/zero_jelly")


    candy_classified = classify_candies(screenshot_path, candies_box, candy_models, candy_class_names, update=False, range1=range1, range2=range2, range3 = range3, range4 = range4)
    gap_classified = [(box, "gap") for box, _ in gap_box]
    objective_classified = classify_candies(screenshot_path, objective_box, objective_models, objective_class_names, update=False, check_candy=False)
    loader_classified = classify_candies(screenshot_path, loader_box, loader_models, loader_class_names, update=False, check_candy=False)

    objective_numbers = get_objective_numbers(screenshot_path, objective_classified)

    print("\nObjectives detected:")
    for idx, (box, label) in enumerate(objective_classified):
        number = objective_numbers[idx][1] if idx < len(objective_numbers) else "?"
        print(f"Objective {idx + 1}: {label} (Number: {number})")
    moves_left = int(detect_moves(screenshot_path))
    print(f"Moves Left: {moves_left}")
    grid = cluster_detections_by_rows(candy_classified, gap_classified, loader_classified, tolerance=40)
    tracker = ObjectivesTracker()
    for i, row in enumerate(grid):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    moves = find_possible_moves(grid)
    print("\nPossible moves:")
    for ((r1, c1), (r2, c2), c1_label, c2_label) in moves:
        print(f"Swap ({r1}, {c1}) [{c1_label[1]}] with ({r2}, {c2}) [{c2_label[1]}]")
    candy_grid, jelly_grid = extract_jelly_grid(grid)
    for i, row in enumerate(candy_grid):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    
    
    objective_targets = {label: int(number) for (_, label), (_, number) in zip(objective_classified, objective_numbers)}
    jelly_grid = infer_hidden_jelly_layers(candy_grid, jelly_grid, objective_targets)
    for i, row in enumerate(jelly_grid):
        print(f"Row {i + 1}: {[jelly_level for jelly_level in row]}")

    best_depth_move, depth_score, depth_tracker = depth_based_simulation(
        candy_grid, jelly_grid, objective_targets, depth=2
    )
    monte_carlo_move, monte_carlo_score, monte_carlo_tracker = monte_carlo_best_move(
        candy_grid, jelly_grid, objective_targets,5)
    # 2. Run Hybrid MCTS with Pruning
    best_mcts_move, mcts_score, mcts_tracker = hybrid_mcts(
        candy_grid, jelly_grid, moves, objective_targets,
        max_depth=3, simulations_per_move=3
    )

    # Print comparison
    print("Move Strategy Comparison")
    print(f"[Depth Search]   Move: {best_depth_move}, Estimated Score: {depth_score:.2f}, Tracker: {depth_tracker}")
    # Now perform the move on the copied grid and print it out to show simulated outcome
    print(f"[Hybrid MCTS]    Move: {best_mcts_move}, Estimated Score: {mcts_score:.2f}, Tracker: {mcts_tracker}")
    print(f"[Monte Carlo]    Move: {monte_carlo_move}, Estimated Score: {monte_carlo_score:.2f}, Tracker: {monte_carlo_tracker}")
    # 3. Run Depth-based Simulation
    steps_taken_depth, depth_tracker, completion, candy_grid1, jelly_grid1, score = simulate_to_completion(candy_grid, jelly_grid, objective_targets, depth_based_simulation, depth = 2, max_steps = moves_left)
    print(f"\nDepth-based Simulation Steps: {steps_taken_depth}")
    print(f"Depth Tracker: {depth_tracker}")
    print(f"Completion Status: {completion}")
    print(f"Score: {score}")
    for i, row in enumerate(candy_grid1):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    for i, row in enumerate(jelly_grid1):
        print(f"Row {i + 1}: {[jelly_level for jelly_level in row]}")
    # 4. Run Monte Carlo Simulation
    steps_taken_monte_carlo, monte_carlo_tracker, completion, candy_grid2, jelly_grid2, score = simulate_to_completion(candy_grid, jelly_grid, objective_targets, monte_carlo_best_move,simulations_per_move = 3, max_steps = moves_left)
    print(f"\nMonte Carlo Simulation Steps: {steps_taken_monte_carlo}")
    print(f"Monte Carlo Tracker: {monte_carlo_tracker}")
    print(f"Completion Status: {completion}")
    print(f"Score: {score}")
    for i, row in enumerate(candy_grid2):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    for i, row in enumerate(jelly_grid2):
        print(f"Row {i + 1}: {[jelly_level for jelly_level in row]}")
    # 5. Run Hybrid MCTS Simulation
    steps_taken_mcts, mcts_tracker, completion, candy_grid3, jelly_grid3, score = simulate_to_completion(candy_grid = candy_grid, jelly_grid = jelly_grid, objective_targets = objective_targets, strategy_fn = hybrid_mcts, max_depth=3, simulations_per_move=3, max_steps = moves_left)
    print(f"\nHybrid MCTS Simulation Steps: {steps_taken_mcts}")    
    print(f"Hybrid MCTS Tracker: {mcts_tracker}")
    print(f"Completion Status: {completion}")
    print(f"Score: {score}")
    for i, row in enumerate(candy_grid3):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    for i, row in enumerate(jelly_grid3):
        print(f"Row {i + 1}: {[jelly_level for jelly_level in row]}")
    steps_taken_expectimax, expectimax_tracker, completion, candy_grid5, jelly_grid5, score = simulate_to_completion(candy_grid = candy_grid, jelly_grid = jelly_grid, objective_targets = objective_targets, strategy_fn = expectimax, max_steps = moves_left)
    print(f"\nExpectimax Simulation Steps: {steps_taken_expectimax}")
    print(f"Expectimax Tracker: {expectimax_tracker}")
    print(f"Completion Status: {completion}")
    print(f"Score: {score}")
    for i, row in enumerate(candy_grid5):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    for i, row in enumerate(jelly_grid5):
        print(f"Row {i + 1}: {[jelly_level for jelly_level in row]}")
    steps_taken_softmax, softmax_tracker, completion, candy_grid6, jelly_grid6, score = simulate_to_completion(candy_grid = candy_grid, jelly_grid = jelly_grid, objective_targets = objective_targets, strategy_fn = heuristics_softmax_best_move, max_steps = moves_left)
    print(f"\nSoftmax Heuristic Simulation Steps: {steps_taken_softmax}")
    print(f"Softmax Heuristic Tracker: {softmax_tracker}")
    print(f"Completion Status: {completion}")
    print(f"Score: {score}")
    for i, row in enumerate(candy_grid6):
        print(f"Row {i + 1}: {[label for _, label in row]}")
    for i, row in enumerate(jelly_grid6):
        print(f"Row {i + 1}: {[jelly_level for jelly_level in row]}")"""
    run_move_selection_experiment(skip_existing_results=True)

    

