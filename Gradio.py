import gradio as gr
import os
from helmet_detect_alert import detect_and_alert, detect_from_images, MODEL_PATHS
from ultralytics import YOLO
from PIL import Image

# Load model
model_path = MODEL_PATHS["YOLOv8n"]
model = YOLO(model_path)

# ---------- Functions ----------
def detect_image_fn(image, confidence):
    if not image:
        return None
    result_path = detect_from_images([image], model, confidence, return_path=True)
    return result_path[0] if result_path else None

def detect_video_fn(video, confidence):
    if not video:
        return None
    output_path = os.path.join("output", os.path.basename(video.name).replace(".", "_pred.")) + "mp4"
    os.makedirs("output", exist_ok=True)
    detect_and_alert(video.name, output_path, model, confidence)
    return output_path

# ---------- Gradio Interfaces ----------
image_interface = gr.Interface(
    fn=detect_image_fn,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(0.1, 1.0, step=0.05, value=0.3, label="Confidence Threshold")
    ],
    outputs="image",
    title="🖼 Helmet Detection - Image",
    description="Upload an image to detect if helmets are worn. Alerts will be shown if heads are not protected."
)

video_interface = gr.Interface(
    fn=detect_video_fn,
    inputs=[
        gr.Video(type="file", label="Upload Video"),
        gr.Slider(0.1, 1.0, step=0.05, value=0.3, label="Confidence Threshold")
    ],
    outputs="video",
    title="🎥 Helmet Detection - Video",
    description="Upload a video and the model will detect people without helmets and generate alerts."
)

# ---------- Tabs ----------
app = gr.TabbedInterface(
    [image_interface, video_interface],
    tab_names=["Image Detection", "Video Detection"]
)

app.launch()
