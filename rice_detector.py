import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("🍚 Broken Rice Detector")
st.write("Upload an image of rice grains to detect broken ones.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def classify_grains(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binary thresholding with Otsu's method
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    # Convert to uint8 and find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # So background is 1, not 0
    markers[unknown == 255] = 0

    # Apply watershed
    image_copy = image.copy()
    markers = cv2.watershed(image_copy, markers)

    output = image.copy()
    broken_count = 0
    total_count = 0

    # Loop over each unique marker
    for marker in np.unique(markers):
        if marker <= 1:
            continue  # Skip background

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            continue

        cnt = cnts[0]
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        total_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        if h < 30 or aspect_ratio > 0.4:
            color = (0, 0, 255)  # Red for broken
            broken_count += 1
        else:
            color = (0, 255, 0)  # Green for whole

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    return output, total_count, broken_count

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    processed_img, total, broken = classify_grains(img_array)
    st.image(processed_img, caption=f"Broken: {broken}, Total: {total}", use_column_width=True)
