import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust the font size of the confusion matrix letters
plt.rcParams.update({'font.size': 16})

# Load the model
model = keras.models.load_model("myModel.h5")

# Define image properties
img_size = (8, 8)
image_folder = "testing"  # Folder containing the images to predict

# Define the class names
# It shall follow the class order that the model learned from
class_names = ['First', 'Second', 'Third', 'Fourth']

# Function to predict a single image
def predict_image(image_path, model, img_size):
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array, verbose=0) # verbose=0 to not print each prediction
    class_index = np.argmax(predictions)
    return class_index

# Collect real classes and predicted classes
true_labels = []    # List to store the real classes
predicted_labels = [] # List to store the predicted classes

print(f"Initializing prediction on the images in the folder: {image_folder}\n")

# The 'image_folder' shall contain subfolders with the name of the classes
# Ex: testing/class_0/img1.png, testing/class_1/img2.png
for class_name in os.listdir(image_folder):
    class_path = os.path.join(image_folder, class_name)
    if os.path.isdir(class_path): # Verify if it is a subfolder (class)
        try:
            # Try to convert the folder name to the class index
            # This will only work when the folders have the names like this: '0', '1', '2', '3'
            # If the folder have different names, it will be necessary a dictionary to map it
            true_class_index = int(class_name) # Or use a dictionary: class_to_idx[class_name]
        except ValueError:
            print(f"Warning: It was not possible to convert '{class_name}' to a numerical index. Verify the folder structure.")
            continue # Skip folder if it is not a numerical index

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                predicted_class_index = predict_image(image_path, model, img_size)

                true_labels.append(true_class_index)
                predicted_labels.append(predicted_class_index)

                print(f"Image: {os.path.join(class_name, image_name)} -> Real: {class_names[true_class_index]} ({true_class_index}), Predicted: {class_names[predicted_class_index]} ({predicted_class_index})")

print(f"\nTotal number of processed images: {len(true_labels)}")

# Generate the confusion matrix
if len(true_labels) == 0:
    print("No image was processed. Impossible to generate confusion matrix.")
else:
    cm = confusion_matrix(true_labels, predicted_labels)
    print("\n--- Confusion Matrix ---")
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='g', # 'g' or 'd' for integers
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
               )
    plt.xlabel('Model Predictions')
    plt.ylabel('Real Classes')
    plt.title('Confusion Matrix')
    plt.show()

    # Detailed classification report
    from sklearn.metrics import classification_report
    print("\n--- Classification Report ---")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))