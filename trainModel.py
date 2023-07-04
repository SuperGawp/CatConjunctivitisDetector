import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the directories for healthy and infected eye images
healthy_dir = './data/healthy'
infected_dir = './data/infected'

# Set the image dimensions
img_width, img_height = 100, 100

# Load the images and labels
healthy_images = []
infected_images = []
labels = []

# Load healthy eye images
for img_name in os.listdir(healthy_dir):
    img = cv2.imread(os.path.join(healthy_dir, img_name))
    img = cv2.resize(img, (img_width, img_height))
    healthy_images.append(img)
    labels.append(0)

# Load infected eye images
for img_name in os.listdir(infected_dir):
    img = cv2.imread(os.path.join(infected_dir, img_name))
    img = cv2.resize(img, (img_width, img_height))
    infected_images.append(img)
    labels.append(1)

# Combine the images and labels
images = np.concatenate((healthy_images, infected_images), axis=0)
labels = np.array(labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the model
model.save('models/model.h5')