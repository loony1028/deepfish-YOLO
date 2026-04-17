import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import time
from collections import Counter

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
    return YOLO("runs/detect/train/weights/best4mseg.pt")

model = load_model()

# -------------------------
# IoU function
# -------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# -------------------------
# Remove duplicate detections
# -------------------------
def filter_duplicates(result, iou_threshold=0.5):
    if result.boxes is None:
        return result

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    keep = []

    for i in range(len(boxes)):
        discard = False
        for j in range(len(boxes)):
            if i != j:
                iou = compute_iou(boxes[i], boxes[j])
                if iou > iou_threshold and scores[i] < scores[j]:
                    discard = True
                    break
        if not discard:
            keep.append(i)

    result.boxes = result.boxes[keep]

    if result.masks is not None:
        result.masks.data = result.masks.data[keep]

    return result


# -------------------------
# Upload image
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # -------------------------
    # Inference + timing
    # -------------------------
    start_time = time.time()

    results = model(
        img_bgr,
        conf=0.1,
        iou=0.5
    )

    result = results[0]

    # Apply duplicate filtering
    result = filter_duplicates(result)

    end_time = time.time()
    inference_time = end_time - start_time

    # Annotated image
    annotated_img = result.plot(conf=False)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # Layout
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detected Output")
        st.image(annotated_img_rgb, use_container_width=True)

    # -------------------------
    # Bottom center: time
    # -------------------------
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 22px; margin-top: 20px;">
        ⏱️ Inference Time: <b>{inference_time:.4f} seconds</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------
    # Detection Details
    # -------------------------
    with st.expander("🔍 Detection Details"):

        if result.boxes is not None:

            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names

            # Total fish count
            total_fish = len(classes)

            # Count per class
            class_counts = Counter(classes)

            st.subheader("📊 Fish Count Summary")

            st.write(f"**Total Fish Detected:** {total_fish}")

            st.write("**Count per Fish Type:**")
            for class_id, count in class_counts.items():
                st.write(f"- {names[class_id]}: {count}")

            # Raw outputs (optional debug)
            # st.subheader("🔬 Raw Model Outputs")
            # st.write("Boxes:", result.boxes.xyxy)
            # st.write("Classes:", result.boxes.cls)
            # st.write("Confidence:", result.boxes.conf)

        else:
            st.write("No fish detected.")

        # if result.masks is not None:
        #     st.write("Mask shape:", result.masks.data.shape)
