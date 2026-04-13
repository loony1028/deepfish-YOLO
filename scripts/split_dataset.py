# scripts/split_dataset.py

import os
import random
from shutil import copy

IMG_DIR = "data/images"
LBL_DIR = "data/labels"

OUT_DIR = "data/yolo"

for split in ["train", "val"]:
    os.makedirs(f"{OUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/{split}/labels", exist_ok=True)

images = os.listdir(IMG_DIR)
random.shuffle(images)

split_idx = int(len(images) * 0.8)

train = images[:split_idx]
val = images[split_idx:]

def copy_files(files, split):
    skipped = 0

    for f in files:
        img_path = f"{IMG_DIR}/{f}"
        label = os.path.splitext(f)[0] + ".txt"
        label_path = f"{LBL_DIR}/{label}"

        # ✅ skip if label doesn't exist
        if not os.path.exists(label_path):
            skipped += 1
            continue

        copy(img_path, f"{OUT_DIR}/{split}/images/{f}")
        copy(label_path, f"{OUT_DIR}/{split}/labels/{label}")

    print(f"{split}: skipped {skipped} images without labels")

copy_files(train, "train")
copy_files(val, "val")

print("✅ Dataset split complete")