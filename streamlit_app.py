import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import gdown

# Title
st.title("🚨 Accident Detection with YOLOv8")

# ✅ Download the model from Google Drive if not present
MODEL_PATH = "best.pt"
GDRIVE_URL = "https://drive.google.com/uc?id=1FBdbIBC7ROxstSFhxveKE0ta13B8y0QV"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Downloading...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# ✅ Load YOLOv8 model
model = YOLO(MODEL_PATH)

# ✅ IMAGE INFERENCE TAB
tab1, tab2 = st.tabs(["🖼️ Image", "🎥 Video"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"Accident: {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Prediction", use_column_width=True)

# ✅ VIDEO INFERENCE TAB
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video")
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        st.write("Processing video...")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for result in results:
                for box in result.boxes:
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
        st.video(output_path)
