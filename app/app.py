# app/app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

st.title("🐟 Fish Detection System")

uploaded_file = st.file_uploader("Upload fish image")

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    results = model(img_array)

    result_img = results[0].plot()

    st.image(result_img, caption="Detected Fish")