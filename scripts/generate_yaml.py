# scripts/generate_yaml.py

import json
import yaml

COCO_JSON = "data/annotations.json"
OUTPUT_YAML = "data/fish.yaml"

with open(COCO_JSON) as f:
    coco = json.load(f)

# Extract class names
categories = coco["categories"]
categories_sorted = sorted(categories, key=lambda x: x["id"])

names = {i: cat["name"] for i, cat in enumerate(categories_sorted)}

data_yaml = {
    "path": "data/yolo",
    "train": "train/images",
    "val": "val/images",
    "names": names
}

with open(OUTPUT_YAML, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("✅ fish.yaml generated successfully!")