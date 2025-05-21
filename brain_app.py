
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

model = load_model("TumorPredictionCNN_model.h5")
IMG_SIZE = 64

st.title("Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    prediction = model.predict(img_input)[0][0]
    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")
    st.write(f"Prediction Confidence: {prediction:.4f}")
