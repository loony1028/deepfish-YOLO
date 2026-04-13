from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")  # start pretrained


model.train(
    data="data/fish.yaml",
    epochs=10,
    device="cpu"
)