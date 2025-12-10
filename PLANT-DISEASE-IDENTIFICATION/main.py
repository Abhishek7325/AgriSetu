import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_plant_disease_model.h5"
IMG_PATH = BASE_DIR / "Diseases.png"

# --- Utility: load model once per app instance ---
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(str(path), compile=False)

# --- Load disease guide (optional) ---
disease_info = {}
md_candidates = [
    BASE_DIR / "DISEASE-GUIDE.md",
    BASE_DIR / "PLANT-DISEASE-IDENTIFICATION" / "DISEASE-GUIDE.md",
    BASE_DIR.parent / "DISEASE-GUIDE.md",
]
md_path = next((p for p in md_candidates if p.exists()), None)
if md_path:
    text = md_path.read_text(encoding="utf-8")
    cur = None
    buf = []
    for line in text.splitlines():
        if line.strip().startswith("###"):
            if cur:
                disease_info[cur] = {"description": "\n".join(buf).strip()}
            cur = line.strip().lstrip("#").strip()
            buf = []
        else:
            if cur is not None:
                buf.append(line)
    if cur:
        disease_info[cur] = {"description": "\n".join(buf).strip()}

# --- Sidebar ---
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# --- Show header image ---
if IMG_PATH.exists():
    img = Image.open(IMG_PATH)
    st.image(img, use_column_width=True)
else:
    st.write("Header image not found at:", IMG_PATH)

# --- Model prediction function ---
def model_prediction(test_image):
    """
    test_image: either a file-like object (UploadedFile) or path-like string
    returns: integer index or None on failure
    """
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

    # load (cached)
    model = load_model(MODEL_PATH)

    # if user didn't upload a file
    if not test_image:
        st.warning("Please upload an image first.")
        return None

    # preprocess and predict
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        return int(np.argmax(predictions))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Main app pages ---
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    st.write("Upload leaf images in the Disease Recognition page to get predictions.")

elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])

    # Preview uploaded image
    if test_image is not None:
        try:
            st.image(test_image, caption="Uploaded image", use_column_width=True)
        except Exception:
            # sometimes UploadedFile is a buffer; reopen via PIL
            img_preview = Image.open(test_image)
            st.image(img_preview, caption="Uploaded image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Running model..."):
            result_index = model_prediction(test_image)

        if result_index is None:
            st.error("Prediction failed or no model available.")
        else:
            # class names list (keep same order as training)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                'Tomato___healthy'
            ]

            predicted_label = class_name[result_index]
            st.success(f"Model predicts: **{predicted_label}**")

            # show description (if available)
            info = disease_info.get(predicted_label)
            if info:
                with st.expander(f"About {predicted_label}"):
                    st.markdown(info.get("description", "No description available."))
            else:
                st.info("No detailed description found for this label.")
