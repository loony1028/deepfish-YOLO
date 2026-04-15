from ultralytics import YOLO

def main():
    # Load YOLOv8 segmentation model (nano version)
    model = YOLO("yolov8n-seg.pt")  # pretrained weights

    # Train the model
    results = model.train(
        data="data/fish.yaml",   # path to dataset config
        epochs=50,              # adjust as needed
        imgsz=640,              # image size
        batch=4,                # keep small for CPU
        device="cpu",           # IMPORTANT: force CPU
        workers=0,              # avoids multiprocessing issues on CPU
        project="runs",         # output folder
        name="fish_seg",        # experiment name
        pretrained=True
    )

if __name__ == "__main__":
    main()