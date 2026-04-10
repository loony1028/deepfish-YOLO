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
    for f in files:
        copy(f"{IMG_DIR}/{f}", f"{OUT_DIR}/{split}/images/{f}")
        label = f.replace(".jpg", ".json")
        copy(f"{LBL_DIR}/{label}", f"{OUT_DIR}/{split}/labels/{label}")

copy_files(train, "train")
copy_files(val, "val")

print("✅ Dataset split complete")