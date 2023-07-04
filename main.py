import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('models/model.h5')

# Set the image dimensions
img_width, img_height = 100, 100

# Make predictions on new images
new_images = []
image_paths = ['./input/healthy1.png']  # Replace with your image paths

for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    new_images.append(img)

new_images = np.array(new_images)
new_images = new_images.astype('float32') / 255.0
predictions = model.predict(new_images)

# Classify the predictions
threshold = 0.5  # Set the threshold probability

for prediction in predictions:
    probability = prediction[0]
    if probability >= threshold:
        print("The eye is infected.")
    else:
        print("The eye is not infected.")
