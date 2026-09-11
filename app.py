
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="✈️",
    layout="centered"
)

st.title("Aerial Object Classification")
st.write("Upload an aerial image to classify it as Bird or Drone.")

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_aerial_classifier.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# --------------------------------------------------
# Load class names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        }
    )

model = load_model()

# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(image_resized)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    probability = float(model.predict(image_array, verbose=0)[0][0])

    # Since bird = 0 and drone = 1
    drone_probability = probability
    bird_probability = 1 - probability

    if drone_probability >= 0.5:
        predicted_class = "drone"
        confidence = drone_probability
    else:
        predicted_class = "bird"
        confidence = bird_probability

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class.upper()}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )

    st.write(
        f"Bird probability: {bird_probability * 100:.2f}%"
    )

    st.write(
        f"Drone probability: {drone_probability * 100:.2f}%"
    )
