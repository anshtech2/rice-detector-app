
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("🍚 Broken Rice Detector")
st.write("Upload an image of rice grains to detect broken ones.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def classify_grains(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    broken_count = 0
    total_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        total_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        if h < 40 or aspect_ratio > 0.7:
            color = (0, 0, 255)
            broken_count += 1
        else:
            color = ( 255,0, 0)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    return output, total_count, broken_count

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    processed_img, total, broken = classify_grains(img_array)
    st.image(processed_img, caption=f"Broken: {broken}, Total: {total}", use_column_width=True)
