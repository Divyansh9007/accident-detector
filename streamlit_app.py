import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import urllib.request
import os
import tempfile

st.set_page_config(page_title="Accident Detection", layout="centered")
st.title("🚨 Accident Detection with YOLOv8")

# --- Download Model from Google Drive if not present ---
MODEL_URL = "https://drive.google.com/uc?id=1FBdbIBC7ROxstSFhxveKE0ta13B8y0QV"
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading YOLO model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded!")

# --- Load YOLO Model on CPU ---
raw_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model = YOLO(raw_model)

# --- Tabs for Image and Video ---
tab1, tab2 = st.tabs(["🖼️ Image Input", "🎥 Video Input"])

# ----------------- IMAGE TAB -----------------
with tab1:
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("🔍 Processing...")

        # Convert to OpenCV
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
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

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Prediction", use_column_width=True)

# ----------------- VIDEO TAB -----------------
with tab2:
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key="video")

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.write("🎬 Processing video...")

        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Output file
        out_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"Accident: {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()

        st.success("✅ Processing complete. Playing output:")
        st.video(out_path)
