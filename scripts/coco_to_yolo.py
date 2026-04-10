# scripts/coco_to_yolo.py

import json
import os
from pathlib import Path
from tqdm import tqdm

# paths
COCO_JSON = "data/coco_format_fish_data.json"
OUTPUT_LABELS = "data/labels"
IMAGES_DIR = "data/images"

os.makedirs(OUTPUT_LABELS, exist_ok=True)

# load json
with open(COCO_JSON) as f:
    coco = json.load(f)

# map image_id → file
images = {img["id"]: img for img in coco["images"]}

# map category_id → class index
categories = coco["categories"]
cat_id_map = {cat["id"]: i for i, cat in enumerate(categories)}

# group annotations by image
annotations = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    annotations.setdefault(img_id, []).append(ann)

# convert
for img_id, anns in tqdm(annotations.items()):
    img_info = images[img_id]
    w, h = img_info["width"], img_info["height"]
    file_name = img_info["file_name"]

    label_file = Path(OUTPUT_LABELS) / (Path(file_name).stem + ".txt")

    with open(label_file, "w") as f:
        for ann in anns:
            cat = cat_id_map[ann["category_id"]]

            # bbox (COCO format: x, y, width, height)
            x, y, bw, bh = ann["bbox"]

            # convert to YOLO
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            f.write(f"{cat} {x_center} {y_center} {bw} {bh}\n")

print("✅ Conversion complete")