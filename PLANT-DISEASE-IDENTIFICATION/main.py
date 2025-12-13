import streamlit as st
import tensorflow as tf
import numpy as np
import re
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


# =====================================================
#     ROBUST DISEASE-GUIDE.md LOADING AND MATCHING
# =====================================================

# Function to normalize label (remove punctuation, spaces, underscores, lowercase)
def normalize_label(text):
    text = text.lower()
    text = re.sub(r'[_\s(),\-]+', '', text)        # remove spaces, underscores, punctuation
    text = re.sub(r'[^a-z0-9]', '', text)          # keep alphanumeric only
    return text

disease_info = {}
md_candidates = [
    BASE_DIR / "DISEASE-GUIDE.md",
    BASE_DIR / "PLANT-DISEASE-IDENTIFICATION" / "DISEASE-GUIDE.md",
    BASE_DIR.parent / "DISEASE-GUIDE.md",
]

md_path = next((p for p in md_candidates if p.exists()), None)

if md_path:
    text = md_path.read_text(encoding="utf-8")

    # Regex to remove numbering like "21." in headings
    header_re = re.compile(r'^\s*#{2,}\s*(?:\d+\.\s*)?(.*)$', flags=re.MULTILINE)

    parts = header_re.split(text)
    # parts = ["before", heading1, text1, heading2, text2, ...]

    it = iter(parts[1:])   # skip "before"
    for heading, content in zip(it, it):
        heading = heading.strip()
        content = content.strip()
        disease_info[heading] = {"description": content}

# Build a normalized lookup index
normalized_index = { normalize_label(k): k for k in disease_info.keys() }


# =====================================================
#           STREAMLIT APP USER INTERFACE
# =====================================================

st.sidebar.title("AgriSetu")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Show header image
if IMG_PATH.exists():
    img = Image.open(IMG_PATH)
    st.image(img, use_column_width=True)
else:
    st.write("Header image not found at:", IMG_PATH)


# --- Model prediction function ---
def model_prediction(test_image):
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

    model = load_model(MODEL_PATH)

    if not test_image:
        st.warning("Please upload an image first.")
        return None

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        return int(np.argmax(predictions))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


# =====================================================
#                     HOME PAGE
# =====================================================

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    st.write("Upload leaf images in the Disease Recognition page to get predictions.")


# =====================================================
#              DISEASE RECOGNITION PAGE
# =====================================================

elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])

    # Preview
    if test_image is not None:
        try:
            st.image(test_image, caption="Uploaded image", use_column_width=True)
        except:
            img_preview = Image.open(test_image)
            st.image(img_preview, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running model..."):
            result_index = model_prediction(test_image)

        if result_index is None:
            st.error("Prediction failed or no model available.")
        else:
            # All class labels
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

            # ===============================
            #    SMART DESCRIPTION MATCHING
            # ===============================
            norm = normalize_label(predicted_label)
            matched_key = normalized_index.get(norm)

            if matched_key:       # exact normalized match
                info = disease_info[matched_key]
                with st.expander(f"About {predicted_label}"):
                    st.markdown(info.get("description", "No description available."))
            else:
                # secondary fuzzy match (optional)
                found = None
                for k in disease_info.keys():
                    if normalize_label(k).find(norm) != -1 or norm.find(normalize_label(k)) != -1:
                        found = k
                        break

                if found:
                    info = disease_info[found]
                    with st.expander(f"About {predicted_label} (matched to {found})"):
                        st.markdown(info.get("description", "No description available."))
                else:
                    st.info("No detailed description found for this label.")

