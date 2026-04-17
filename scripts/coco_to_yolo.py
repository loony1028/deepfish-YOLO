# convert_to_yolo_bbox.py

import json
import os
import cv2

INPUT_JSON = "coco_annotations.json"
IMAGE_DIR = "images/"
OUTPUT_LABELS = "labels/"

os.makedirs(OUTPUT_LABELS, exist_ok=True)

with open(INPUT_JSON) as f:
    data = json.load(f)

for item in data:
    image_path = os.path.join(IMAGE_DIR, item["file_name"])
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    label_file = os.path.join(OUTPUT_LABELS, item["file_name"].replace(".jpg", ".txt"))

    with open(label_file, "w") as f:
        for ann in item["annotations"]:
            class_id = ann["category_id"]
            x, y, bw, bh = ann["bbox"]

            # convert to YOLO format
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            bw /= w
            bh /= h

            f.write(f"{class_id} {xc} {yc} {bw} {bh}\n")