import cv2
import torch
import streamlit as st
from ultralytics import YOLO
import numpy
# Cache model supaya tidak dimuat ulang
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # Bisa ganti ke 's', 'm', 'l', 'x' sesuai kebutuhan
    model.to("cpu")  # Paksa ke CPU agar kompatibel di semua perangkat
    return model

model = load_model()

def detect_human(video_url):
    cap = cv2.VideoCapture(video_url)
    
    # Cek apakah stream terbuka
    if not cap.isOpened():
        st.error("Gagal membuka video stream! Periksa URL atau koneksi jaringan.")
        return
    
    stframe = st.empty()
    
    while st.session_state["detecting"]:  # Gunakan session_state agar bisa dihentikan
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
    st.info("Deteksi dihentikan.")

# Streamlit UI
st.title("Deteksi Manusia dengan YOLOv8")

# Input URL IP Webcam
url = st.text_input("Masukkan URL IP Webcam (contoh: http://192.168.1.5:8080/video)")

# Inisialisasi session_state jika belum ada
if "detecting" not in st.session_state:
    st.session_state["detecting"] = False

# Tombol mulai deteksi
if st.button("Mulai Deteksi"):
    if url:
        st.session_state["detecting"] = True
        detect_human(url)
    else:
        st.warning("Masukkan URL terlebih dahulu!")

# Tombol stop deteksi
if st.button("Stop Deteksi"):
    st.session_state["detecting"] = False
