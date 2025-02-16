import cv2
import torch
import streamlit as st
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8n.pt")  # Bisa ganti ke 's', 'm', 'l', 'x' sesuai kebutuhan
model.to("cpu")

def detect_human(video_url, stop_flag):
    cap = cv2.VideoCapture(video_url)
    stframe = st.empty()
    
    while cap.isOpened():
        if stop_flag():
            break
        
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal mendapatkan video stream. Periksa IP atau koneksi jaringan.")
            break
        
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label != "person":
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                conf = box.conf[0].item()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        stframe.image(frame, channels="BGR")
        
    cap.release()

# Streamlit UI
st.title("Deteksi Manusia dengan YOLOv8")
url = st.text_input("Masukkan URL IP Webcam (contoh: http://192.168.1.5:8080/video)")

start_detection = st.button("Mulai Deteksi")
stop_detection = st.button("Stop Deteksi")

def stop_flag():
    return stop_detection

if start_detection:
    if url:
        detect_human(url, stop_flag)
    else:
        st.warning("Masukkan URL terlebih dahulu!")
