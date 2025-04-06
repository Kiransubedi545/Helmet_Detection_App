import gradio as gr
import os
from helmet_detect_alert import detect_and_alert, detect_from_images, MODEL_PATHS
from ultralytics import YOLO

# Load YOLOv8 model
model_path = MODEL_PATHS["YOLOv8n"]
model = YOLO(model_path)

# ---------- Detection Functions ----------
def detect_image_fn(image_path, confidence):
    if image_path is None:
        return None
    results = detect_from_images([image_path], model, confidence, return_path=True)
    return results[0] if results else None

def detect_video_fn(video_file, confidence):
    if video_file is None:
        return None
    input_path = video_file
    output_path = os.path.join("output", os.path.basename(input_path).replace(".", "_pred.")) + "mp4"
    os.makedirs("output", exist_ok=True)
    detect_and_alert(input_path, output_path, model, confidence)
    return output_path

# ---------- Interfaces ----------
image_interface = gr.Interface(
    fn=detect_image_fn,
    inputs=[
        gr.Image(label="Upload Image"),  # No type="filepath"
        gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Confidence")
    ],
    outputs=gr.Image(label="Predicted Output"),
    title="🖼 Helmet Detection from Image",
    description="Upload an image to detect heads not wearing helmets."
)

video_interface = gr.Interface(
    fn=detect_video_fn,
    inputs=[
        gr.Video(label="Upload Video"),  # No type="file"
        gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Confidence")
    ],
    outputs=gr.Video(label="Predicted Output"),
    title="🎥 Helmet Detection from Video",
    description="Upload a video and get helmet detection alerts visually."
)

# ---------- Launch Tabbed App ----------
gr.TabbedInterface(
    [image_interface, video_interface],
    tab_names=["Image Detection", "Video Detection"]
).launch(share=True)
