import streamlit as st
from ultralytics import YOLO
from helmet_detect_alert import detect_from_images, detect_and_alert, MODEL_PATHS

# Load model
model = YOLO(MODEL_PATHS["YOLOv8n"])

st.title("🪖 Helmet Detection Web App")
st.markdown("Upload images or videos to detect people without helmets.")

option = st.radio("Choose Input Type", ["Image", "Video"])

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

if option == "Image":
    uploaded_images = st.file_uploader("Upload one or more images", accept_multiple_files=True, type=["jpg", "png"])
    if st.button("Detect in Images") and uploaded_images:
        for image_file in uploaded_images:
            with open(image_file.name, "wb") as f:
                f.write(image_file.read())
            detect_from_images([image_file.name], model, confidence)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_video and st.button("Detect in Video"):
        with open(uploaded_video.name, "wb") as f:
            f.write(uploaded_video.read())
        detect_and_alert(uploaded_video.name, model, confidence)

