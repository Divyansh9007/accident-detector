import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("🚨 Accident Detection with YOLOv8")

# Load YOLO model
model = YOLO("best.pt")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Inference
    results = model(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Accident: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show result
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Prediction", use_column_width=True)
