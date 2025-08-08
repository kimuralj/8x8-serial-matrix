import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("myModel.h5")

# Define image properties
img_size = (8, 8)
image_folder = "testing"  # Folder containing images to predict

# Function to predict a single image
def predict_image(image_path, model):
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return class_index

# Predict all images in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):  # Ensure it's an image
        predicted_class = predict_image(image_path, model)
        print(f"{image_name} -> Predicted class: CLASS_{predicted_class}")
