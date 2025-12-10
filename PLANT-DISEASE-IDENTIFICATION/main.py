import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Paths & model loading
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_FILENAME = "trained_plant_disease_model.keras"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Attempt to load the model once (safer and faster than loading per request)
model = None
if os.path.exists(MODEL_PATH):
    try:
        # compile=False avoids errors related to missing optimizer states / custom objects
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        # Keep model as None so UI can show a helpful message
else:
    st.sidebar.warning(f"Model not found at {MODEL_PATH}. Please upload the model to the repo or provide a download mechanism.")

# -------------------------
# Prediction function
# -------------------------
def model_prediction_from_pil(pil_image):
    """
    Accepts a PIL.Image object, preprocesses and predicts.
    Returns integer index of predicted class.
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    # Ensure RGB and correct size
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((128, 128))

    input_arr = tf.keras.preprocessing.image.img_to_array(pil_image)
    input_arr = np.expand_dims(input_arr, axis=0)  # shape: (1, 128, 128, 3)

    predictions = model.predict(input_arr)
    return int(np.argmax(predictions, axis=1)[0])

# -------------------------
# Helper: read uploaded file to PIL
# -------------------------
def uploadedfile_to_pil(uploaded_file):
    """
    uploaded_file: Streamlit UploadedFile or path string
    returns PIL.Image
    """
    if uploaded_file is None:
        return None
    # If it's an UploadedFile (in-memory), open directly with PIL
    if hasattr(uploaded_file, "read"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return Image.open(uploaded_file)
    # if a filesystem path was passed for testing
    return Image.open(uploaded_file)

# -------------------------
# UI: show header image
# -------------------------
# Show banner image (ensure correct path)
banner_path = os.path.join(BASE_DIR, "Diseases.png")
if os.path.exists(banner_path):
    try:
        banner_img = Image.open(banner_path)
        st.image(banner_img, use_column_width=True)
    except Exception:
        # If banner fails to open, ignore it
        pass

# -------------------------
# Main UI
# -------------------------
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    st.write("Use the sidebar to go to DISEASE RECOGNITION and upload an image to predict.")

elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

    if test_image is not None:
        # Display the uploaded image
        try:
            pil_img = uploadedfile_to_pil(test_image)
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Could not open uploaded image: {e}")
            pil_img = None
    else:
        pil_img = None

    # Predict button
    if st.button("Predict"):
        if model is None:
            st.error("Model is not loaded. Please ensure the model file 'trained_plant_disease_model.keras' is present in the same folder as this script.")
        elif pil_img is None:
            st.error("Please upload an image before clicking Predict.")
        else:
            with st.spinner("Model is predicting..."):
                try:
                    result_index = model_prediction_from_pil(pil_img)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    # Class labels (as you had)
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']

                    if 0 <= result_index < len(class_name):
                        st.success(f"Model predicts: **{class_name[result_index]}**")
                    else:
                        st.error(f"Prediction index out of range: {result_index}")

from pathlib import Path
import re

# LOAD DISEASE GUIDE -----
BASE_PATH = Path(__file__).resolve().parent

# Try to find DISEASE-GUIDE.md in both root and subfolder
possible_paths = [
    BASE_PATH / "DISEASE-GUIDE.md",
    BASE_PATH.parent / "DISEASE-GUIDE.md",
    BASE_PATH / "PLANT-DISEASE-IDENTIFICATION" / "DISEASE-GUIDE.md",
]

disease_guide_path = None
for p in possible_paths:
    if p.exists():
        disease_guide_path = p
        break

disease_info = {}

if disease_guide_path:
    text = disease_guide_path.read_text(encoding="utf-8")
    current = None
    buf = []
    for line in text.splitlines():
        if line.startswith("###"):   # disease heading
            if current:
                disease_info[current] = "\n".join(buf).strip()
            current = line.replace("###", "").strip()
            buf = []
        else:
            if current:
                buf.append(line)
    if current:
        disease_info[current] = "\n".join(buf).strip()
else:
    print("❌ DISEASE-GUIDE.md not found!")

predicted = predicted_class  # model output

# Try to fetch description
description = disease_info.get(predicted)

import streamlit as st

if description:
    with st.expander(f"About {predicted}"):
        st.markdown(description)
else:
    st.warning("No description found for this disease.")


