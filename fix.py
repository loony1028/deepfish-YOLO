import os
from pathlib import Path

# ===== CONFIG =====
DATASET_PATH = "data/yolo/train"  # change if needed
REMOVE_IMAGES_WITH_BAD_LABELS = True  # set False if you want to keep images

LABELS_DIR = Path(DATASET_PATH) / "labels"
IMAGES_DIR = Path(DATASET_PATH) / "images"

# ===== HELPERS =====
def is_valid_polygon(parts):
    """
    YOLO segmentation format:
    class_id x1 y1 x2 y2 x3 y3 ...
    -> need at least 3 points => 6 coords + 1 class = 7 values
    """
    if len(parts) < 7:
        return False

    try:
        coords = list(map(float, parts[1:]))

        # must be even number of coords (x,y pairs)
        if len(coords) % 2 != 0:
            return False

        # values must be between 0 and 1
        for v in coords:
            if v < 0 or v > 1:
                return False

    except:
        return False

    return True


def get_image_path(label_path):
    """Match label file to image file"""
    base = label_path.stem
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = IMAGES_DIR / (base + ext)
        if img_path.exists():
            return img_path
    return None


# ===== MAIN CLEANING =====
bad_files = 0
fixed_files = 0
removed_images = 0

print(f"Scanning labels in: {LABELS_DIR}\n")

for label_file in LABELS_DIR.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        print(f"[EMPTY] {label_file}")
        bad_files += 1

        if REMOVE_IMAGES_WITH_BAD_LABELS:
            img = get_image_path(label_file)
            if img and img.exists():
                img.unlink()
                removed_images += 1
                print(f"   -> removed image {img}")

        label_file.unlink()
        continue

    valid_lines = []

    for line in lines:
        parts = line.strip().split()

        if is_valid_polygon(parts):
            valid_lines.append(line)
        else:
            print(f"[INVALID] {label_file} -> {line.strip()}")

    if len(valid_lines) == 0:
        print(f"[NO VALID OBJECTS] {label_file}")
        bad_files += 1

        if REMOVE_IMAGES_WITH_BAD_LABELS:
            img = get_image_path(label_file)
            if img and img.exists():
                img.unlink()
                removed_images += 1
                print(f"   -> removed image {img}")

        label_file.unlink()
        continue

    # rewrite cleaned label file
    if len(valid_lines) != len(lines):
        with open(label_file, "w") as f:
            f.writelines(valid_lines)
        fixed_files += 1

print("\n===== SUMMARY =====")
print(f"Bad label files removed: {bad_files}")
print(f"Label files fixed: {fixed_files}")
print(f"Images removed: {removed_images}")
print("Done.")