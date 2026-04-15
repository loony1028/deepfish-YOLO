import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Fish Detection App", layout="wide")

st.title("🐟 Fish Segmentation AI App")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best2.pt")

model = load_model()

# -------------------------
# Upload image
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert RGB → BGR for YOLO/OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # -------------------------
    # Run inference + timing
    # -------------------------
    start_time = time.time()

    results = model(img_bgr, conf = 0.2)

    end_time = time.time()
    inference_time = end_time - start_time

    result = results[0]

    # Get annotated image (boxes + masks)
    annotated_img = result.plot()

    # Convert back to RGB for Streamlit display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # Layout: Left + Right
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detected Output")
        st.image(annotated_img_rgb, use_container_width=True)

    # -------------------------
    # Bottom center: inference time
    # -------------------------
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 20px; margin-top: 20px;">
        ⏱️ Inference Time: <b>{inference_time:.4f} seconds</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------
    # Extra debug info (optional)
    # -------------------------
    # with st.expander("🔍 Detection Details"):
    #     if result.boxes is not None:
    #         st.write("Boxes:", result.boxes.xyxy)
    #         st.write("Classes:", result.boxes.cls)
    #         st.write("Confidence:", result.boxes.conf)

    #     if result.masks is not None:
    #         st.write("Mask shape:", result.masks.data.shape)
            