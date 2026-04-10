from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # start pretrained

model.train(
    data="data/fish.yaml",
    epochs=100,
    imgsz=1024,
    batch=8,
    device=0
)