import tensorflow as tf
import numpy as np
import cv2
import sys

# Load trained CNN model
model = tf.keras.models.load_model("deepfake_cnn.h5")

# Load and preprocess the input image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        print(f"🔴 Fake Image (Confidence: {prediction:.2f})")
    else:
        print(f"🟢 Real Image (Confidence: {1 - prediction:.2f})")

# Run the script with an image argument
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_cnn.py <image_path>")
    else:
        predict_image(sys.argv[1])
