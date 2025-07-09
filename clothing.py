import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_classification_model(model_path):
    """Loads the saved Keras model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(150, 150)):
    """Loads and preprocesses an image for model prediction."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Rescale pixel values
    return img_array

if __name__ == '__main__':
    # Example usage (replace with your model path)
    model_path = 'clothing_classification_model.h5'
    loaded_model = load_classification_model(model_path)

    if loaded_model:
        loaded_model.summary()
        # You can now use loaded_model for predictions
        # For example:
        # dummy_image = np.zeros((1, 150, 150, 3)) # Replace with actual image loading
        # predictions = loaded_model.predict(dummy_image)
        # print(predictions)