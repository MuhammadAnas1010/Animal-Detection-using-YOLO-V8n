import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Load the trained model
model = YOLO('best (2).pt')  # Path to your trained YOLOv8 model

st.title("Live YOLOv8 Object Detection")
run = st.checkbox('Start Webcam Detection')
FRAME_WINDOW = st.image([])

# Capture webcam
camera = cv2.VideoCapture(0)

if run:
    while True:
        success, frame = camera.read()
        if not success:
            st.error("Failed to read from webcam.")
            break

        # Resize for performance (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Run detection
        results = model(frame_resized, conf=0.3)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Show in Streamlit
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

        # Control frame rate (reduce CPU load)
        time.sleep(0.03)
else:
    st.write('Webcam stopped.')
    camera.release()
