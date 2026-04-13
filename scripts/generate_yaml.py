# scripts/generate_yaml.py

import json
import yaml

COCO_JSON = "data/coco_format_fish_data.json"
OUTPUT_YAML = "data/fish.yaml"

with open(COCO_JSON) as f:
    coco = json.load(f)

# sort categories by id (VERY IMPORTANT)
categories = sorted(coco["categories"], key=lambda x: x["id"])

names = {i: cat["name"] for i, cat in enumerate(categories)}

data = {
    "path": "data/yolo",
    "train": "train/images",
    "val": "val/images",
    "names": names
}

with open(OUTPUT_YAML, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("✅ fish.yaml generated correctly")