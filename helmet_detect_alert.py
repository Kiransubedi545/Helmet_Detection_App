import cv2
import os
import pyttsx3
from ultralytics import YOLO
import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import csv
from gtts import gTTS
import tempfile
import pygame
from datetime import datetime

# ----------------- Configuration -----------------
DOWNLOAD_PATH = r"D:\Learning Programming\Python\Deep learning\YOLOv8\Test_videos"
MODEL_PATHS = {
    "YOLOv8n": r"D:\Learning Programming\Python\Deep learning\YOLOv8\runs\detect\train5\weights\best.pt"
}
ALERT_ICON_PATH = "alert_icon.png"
ALERT_LOG_PATH = "alert_log.csv"
CONFIDENCE_THRESHOLD = 0.3
ALERT_LANGUAGE = "en"

# ----------------- Initialize Modules -----------------
pygame.init()
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# ----------------- Add Alert Symbol -----------------
def overlay_alert_icon(frame, x, y):
    if not os.path.exists(ALERT_ICON_PATH):
        return frame
    icon = cv2.imread(ALERT_ICON_PATH, cv2.IMREAD_UNCHANGED)
    if icon is None:
        return frame
    icon = cv2.resize(icon, (50, 50))
    ih, iw = icon.shape[:2]
    if y+ih > frame.shape[0] or x+iw > frame.shape[1]:
        return frame
    alpha_s = icon[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        frame[y:y+ih, x:x+iw, c] = (alpha_s * icon[:, :, c] + alpha_l * frame[y:y+ih, x:x+iw, c])
    return frame

# ----------------- Voice Alert -----------------
def speak_alert(text, lang="en"):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text, lang=lang)
        temp_path = f"{fp.name}.mp3"
        tts.save(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

# ----------------- Save Alert Log -----------------
def log_alert(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ALERT_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, message])

# ----------------- Run YOLO Prediction + Alerts -----------------
def detect_and_alert(video_path, model, confidence=CONFIDENCE_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    alert_cooldown = 15
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        results = model.predict(source=frame, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()
        helmets, heads = [], []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if cls_id == 0:
                helmets.append(((x1, y1, x2, y2), (center_x, center_y)))
            elif cls_id == 2:
                heads.append(((x1, y1, x2, y2), (center_x, center_y)))
        alert_triggered = False
        for (hx1, hy1, hx2, hy2), (hcx, hcy) in heads:
            covered = any(abs(hcx - hel_x) < 30 and abs(hcy - hel_y) < 30 for (_, _, _, _), (hel_x, hel_y) in helmets)
            if not covered:
                annotated_frame = overlay_alert_icon(annotated_frame, hx1, hy1)
                cv2.putText(annotated_frame, "⚠ No Helmet!", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_triggered = True
                log_alert(f"Alert at frame {frame_count}: No helmet at ({hx1}, {hy1})")
        if alert_triggered and frame_count % alert_cooldown == 0:
            speak_alert("Alert! Person without helmet detected", ALERT_LANGUAGE)
        cv2.imshow("Helmet Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------- Detect from Multiple Images -----------------
def detect_from_images(image_paths, model, confidence=CONFIDENCE_THRESHOLD):
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        results = model.predict(source=frame, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()
        helmets, heads = [], []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if cls_id == 0:
                helmets.append(((x1, y1, x2, y2), (center_x, center_y)))
            elif cls_id == 2:
                heads.append(((x1, y1, x2, y2), (center_x, center_y)))
        alert_triggered = False
        for (hx1, hy1, hx2, hy2), (hcx, hcy) in heads:
            covered = any(abs(hcx - hel_x) < 30 and abs(hcy - hel_y) < 30 for (_, _, _, _), (hel_x, hel_y) in helmets)
            if not covered:
                annotated_frame = overlay_alert_icon(annotated_frame, hx1, hy1)
                cv2.putText(annotated_frame, "⚠ No Helmet!", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_triggered = True
                log_alert(f"Alert: No helmet at ({hx1}, {hy1}) in image {image_path}")
        if alert_triggered:
            speak_alert("Alert! Person without helmet detected", ALERT_LANGUAGE)
        cv2.imshow(f"Result - {os.path.basename(image_path)}", annotated_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------- Snapshot -----------------
def save_snapshot(frame, path="snapshot.jpg"):
    cv2.imwrite(path, frame)

# ----------------- Main Execution -----------------
model = YOLO(MODEL_PATHS["YOLOv8n"])
# This script is now module-only: import and use functions from app.py
