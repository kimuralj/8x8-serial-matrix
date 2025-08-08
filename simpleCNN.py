import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
dataset_path = "Dataset"

# Image parameters
img_size = (8, 8)
batch_size = 32
epochs = 30

# Load data using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Define a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(8, 8, 3)),
    layers.MaxPooling2D((2,2), padding="same"),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(4, activation="softmax")  # 4 classes
])

# Compile model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save the model
model.save("myModel.h5")

# Load and predict on a new image
def predict_image(image_path, model):
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return class_index

# Load the trained model
model = keras.models.load_model("myModel.h5")

# # Example prediction
# image_path = "new_image.jpg"
# predicted_class = predict_image(image_path, model)
# print(f"Predicted class: CLASS_{predicted_class + 1}")
