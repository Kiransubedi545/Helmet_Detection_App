import gradio as gr
import cv2
import os
from ultralytics import YOLO
from helmet_detect_alert import detect_from_images, detect_and_alert, MODEL_PATHS, CONFIDENCE_THRESHOLD

model_cache = {}

def load_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = YOLO(MODEL_PATHS[model_name])
    return model_cache[model_name]

def process_image(image, model_name, confidence):
    model = load_model(model_name)
    results = model.predict(source=image, conf=confidence, verbose=False)
    annotated = results[0].plot()
    return annotated

def process_video(video, model_name, confidence):
    input_path = "temp_input_video.mp4"
    output_path = "temp_output_video.mp4"
    with open(input_path, "wb") as f:
        f.write(video.read())
    model = load_model(model_name)
    detect_and_alert(input_path, output_path, model, confidence)
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("# 🪖 Helmet Detection App (YOLOv8)")

    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_output = gr.Image(label="Detection Result")

        with gr.Row():
            conf_slider_img = gr.Slider(0.1, 1.0, value=CONFIDENCE_THRESHOLD, label="Confidence Threshold")
            model_selector_img = gr.Dropdown(choices=list(MODEL_PATHS.keys()), value="YOLOv8n", label="YOLO Model")
            btn_detect_img = gr.Button("🔍 Detect Helmet")

        btn_detect_img.click(fn=process_image, inputs=[image_input, model_selector_img, conf_slider_img], outputs=image_output)

    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            video_input = gr.File(file_types=[".mp4"], label="Upload Video")
            video_output = gr.Video(label="Detection Result")

        with gr.Row():
            conf_slider_vid = gr.Slider(0.1, 1.0, value=CONFIDENCE_THRESHOLD, label="Confidence Threshold")
            model_selector_vid = gr.Dropdown(choices=list(MODEL_PATHS.keys()), value="YOLOv8n", label="YOLO Model")
            btn_detect_vid = gr.Button("🎬 Detect Helmet in Video")

        btn_detect_vid.click(fn=process_video, inputs=[video_input, model_selector_vid, conf_slider_vid], outputs=video_output)

    gr.Markdown("---")
    gr.Markdown("Developed by **Kiran Subedi** | kiransubedi545@gmail.com | [Website](http://kiransubedi545.com.np)")

if __name__ == "__main__":
    demo.launch()