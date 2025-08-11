import os

# === CONFIGURATION ===
labels_dir = "data/labels/train"
start_file = "board9.txt"  # begin remapping from this file onward
remap = {
    0: 1,
    1: 2,
    2: 0
}

# Process all .txt files in lexicographic order
for file_name in sorted(os.listdir(labels_dir)):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, file_name)

    if file_name == start_file:
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            new_class_id = remap.get(class_id, class_id)
            new_line = f"{new_class_id} " + " ".join(parts[1:])
            new_lines.append(new_line)

        with open(file_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f" Remapped: {file_name}")
    else:
        print(f"Skipped:  {file_name}")
