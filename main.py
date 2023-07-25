import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('models/model.h5')

# Set the image dimensions
img_width, img_height = 100, 100

new_images = []
# input the image you want to test
image_paths = ['./input/healthy3.png']  

for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    new_images.append(img)

new_images = np.array(new_images)
new_images = new_images.astype('float32') / 255.0
predictions = model.predict(new_images)

# Set the threshold probability
threshold = 0.5  

for i, prediction in enumerate(predictions):
    probability = prediction[0]
    if probability >= threshold:
        print(f"Image {i+1}: The eye is infected with a probability of {probability:.2f}.")
    else:
        print(f"Image {i+1}: The eye is not infected with a probability of {1 - probability:.2f}.")
