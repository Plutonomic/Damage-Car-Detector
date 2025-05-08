# app.py

import streamlit as st
import cv2
import numpy as np
import os
from cnn import CarDamageClassifier
from preprocessing import resize_image, convert_to_hsv, apply_threshold

# Set up Streamlit
st.set_page_config(page_title="🚗 Damaged Car Image Preprocessing", layout="centered")
st.title("🚗 Damaged Car Image Preprocessing Pipeline")
st.write("Upload a damaged car image for preprocessing and to classify whether it's damaged or not.")

# Load the trained model
classifier = CarDamageClassifier()
classifier.load_model("car_damage_model.h5")

# Image upload functionality
image = None
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Preprocess and classify the uploaded image
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Predict if the car is damaged or not
    result = classifier.predict(image)
    st.subheader(f"Prediction: The car is {result}")

    # Preprocessing steps (optional for EDA)
    resized = resize_image(image)
    hsv = convert_to_hsv(resized)
    thresholded = apply_threshold(hsv)

    st.subheader("Preprocessed Image")
    st.image(thresholded, caption='Thresholded Image', use_container_width=True, channels='GRAY')

    # Save processed image
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "processed_image.png")
    cv2.imwrite(output_path, thresholded)
    st.success(f"Preprocessed image saved to {output_path}")
