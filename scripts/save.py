def save_yolo_segmentation(label_path, polygons, class_id, w, h):
    with open(label_path, "a") as f:
        normalized = []
        for x, y in polygons:
            normalized.append(x / w)
            normalized.append(y / h)

        line = f"{class_id} " + " ".join(map(str, normalized))
        f.write(line + "\n")