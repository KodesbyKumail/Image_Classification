import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Define the target size for the images (should match the model's input size)
TARGET_SIZE = (150, 150)
# Replace with the actual path to your saved model file in your PyCharm project
MODEL_PATH = 'clothing_classification_model.h5'

# Define the class names in the same order as during training
# IMPORTANT: Make sure this list matches the order of classes in your training data
CLASS_NAMES = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']


# Load the model (cache it for efficiency)
@st.cache_resource
def load_classification_model(model_path):
    """Loads the saved Keras model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image_file, target_size=TARGET_SIZE):
    """Loads and preprocesses an image file for model prediction."""
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Rescale pixel values
    return img_array

# Load the model
model = load_classification_model(MODEL_PATH)

if model is not None:
    st.title("Clothing Image Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(uploaded_file)
        predictions = model.predict(processed_image)

        # Get the predicted class name
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        st.write(f"Prediction: **{predicted_class_name}**")

else:
    st.error("Model could not be loaded. Please check the model path.")