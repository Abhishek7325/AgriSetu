# convert_to_h5.py
import tensorflow as tf
from pathlib import Path

# The folder where convert_to_h5.py is located
BASE = Path(__file__).resolve().parent

# Correct path to your model file
keras_path = BASE / "trained_plant_disease_model.keras"
h5_path = BASE / "trained_plant_disease_model.h5"

print("Looking for model at:", keras_path)

if not keras_path.exists():
    print("❌ ERROR: .keras model file NOT FOUND at:", keras_path)
    print("Make sure convert_to_h5.py is in the SAME folder as trained_plant_disease_model.keras")
    exit()

print("✔ .keras model found. Loading model...")

model = tf.keras.models.load_model(str(keras_path), compile=False)

print("✔ Model loaded successfully.")
print("Saving as .h5 ...")

model.save(str(h5_path), include_optimizer=False)

print("🎉 Conversion successful!")
print("Saved H5 model at:", h5_path)
