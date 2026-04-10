# scripts/infer.py

from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

img = cv2.imread("test.jpg")

results = model(img)

output = results[0].plot()

cv2.imwrite("output.jpg", output)