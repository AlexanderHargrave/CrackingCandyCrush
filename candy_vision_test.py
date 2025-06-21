import os
import json
import torch
from candy_vision_train import (
    load_models_for_task,
    detect_candies_yolo,
    classify_candies,
    extract_jelly_colour_range,
    get_objective_numbers,
    detect_moves,
    cluster_detections_by_rows,
)

yolo_model_path = "runs/detect/train7/weights/best.pt"
sample_eval_size = 1
num_epochs = 50
model_names = ["efficientnet_b0", "efficientnet_b3", "resnet18", "resnet34", "resnet50"]
short_model_names = ["efficientnet_b0", "resnet18", "resnet34"]
test_image_dir = "data/test/images"
label_dir = "data/test/labels"

print("Loading models...")
candy_models, candy_class_names = load_models_for_task(
    task_name="candy",
    data_dir="candy_dataset",
    model_names=model_names,
    num_epochs=num_epochs,
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

range1 = extract_jelly_colour_range("jelly_levels/one_jelly")
range2 = extract_jelly_colour_range("jelly_levels/two_jelly")

def load_ground_truth(label_path):
    with open(label_path, 'r') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"Label file {label_path} is empty.")
        return json.loads(content)
def assert_test(image_path, label_path, range1 = range1, range2 = range2):
    gt = load_ground_truth(label_path)

    # Detection
    candies_box, gap_box, loader_box, objective_box = detect_candies_yolo(image_path, yolo_model_path)
    # Classification
    candy_classified = classify_candies(image_path, candies_box, candy_models, candy_class_names, update=False, range1=range1, range2=range2, check_candy=True)
    gap_classified = [(box, "gap") for box, _ in gap_box]
    objective_classified = classify_candies(image_path, objective_box, objective_models, objective_class_names, update=False, check_candy=False)
    loader_classified = classify_candies(image_path, loader_box, loader_models, loader_class_names, update=False, check_candy=False)

    objective_numbers = get_objective_numbers(image_path, objective_classified)
    moves_left = int(detect_moves(image_path))
    grid = cluster_detections_by_rows(candy_classified, gap_classified, loader_classified, tolerance=40)
    # === Assertions ===
    assert moves_left == gt["moves_left"], f"Moves mismatch: got {moves_left}, expected {gt['moves_left']}"

    # Objectives
    for idx, (box, label) in enumerate(objective_classified):
        number = objective_numbers[idx][1] if idx < len(objective_numbers) else "?"
        number = int(number) if number != "?" else -1  # Convert to int or -1 if unknown
        expected_label = gt["objectives"][idx]["label"]
        expected_number = int(gt["objectives"][idx]["number"])
        assert label == expected_label, f"Objective label mismatch at index {idx}: got {label}, expected {expected_label}"
        assert number == expected_number, f"Objective number mismatch at index {idx}: got {number}, expected {expected_number}"

    # Grid shape
    assert len(grid) == gt["grid_rows"], f"Grid row count mismatch: got {len(grid)}, expected {gt['grid_rows']}"
    assert len(grid[0]) == gt["grid_columns"], f"Grid column count mismatch: got {len(grid[0])}, expected {gt['grid_columns']}"

    # Candies (optionally check labels)
    detected_labels = [label for row in grid for _, label in row]
    expected_labels = gt["grid_labels"]
    assert detected_labels == expected_labels, f"Grid labels mismatch."
    print(f"✅ {os.path.basename(image_path)} passed.")

def run_all_tests():
    test_files = [f for f in os.listdir(test_image_dir) if f.endswith(".png") or f.endswith(".jpg")]

    for test_img in test_files:
        test_img_path = os.path.join(test_image_dir, test_img)
       
        test_label_path = os.path.join(label_dir, os.path.splitext(test_img)[0] + ".json")
        assert os.path.exists(test_label_path), f"No label file found for {test_img}"
        try:
            assert_test(test_img_path, test_label_path)
        except AssertionError as e:
            print(f"❌ {test_img} failed: {e}")

if __name__ == "__main__":
    run_all_tests()