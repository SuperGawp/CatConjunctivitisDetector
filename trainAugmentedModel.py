import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define the directories for healthy and infected eye images
healthy_dir = './data/healthy'
infected_dir = './data/infected'

# Set the image dimensions
img_width, img_height = 100, 100

# Load the original images and labels
original_healthy_images = []
original_infected_images = []
labels = []

# Load healthy eye images
for img_name in os.listdir(healthy_dir):
    img = cv2.imread(os.path.join(healthy_dir, img_name))
    img = cv2.resize(img, (img_width, img_height))
    original_healthy_images.append(img)
    labels.append(0)

# Load infected eye images
for img_name in os.listdir(infected_dir):
    img = cv2.imread(os.path.join(infected_dir, img_name))
    img = cv2.resize(img, (img_width, img_height))
    original_infected_images.append(img)
    labels.append(1)

# Combine the original images and labels
original_images = np.concatenate((original_healthy_images, original_infected_images), axis=0)
labels = np.array(labels)

# Split the dataset into training and test sets
X_train_original, X_test, y_train_original, y_test = train_test_split(original_images, labels, test_size=0.2, random_state=42)

# Normalize pixel values between 0 and 1
X_train_original = X_train_original.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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

# Train the model on the original dataset
model.fit(X_train_original, y_train_original, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# Train the model on the augmented dataset
datagen.fit(X_train_original)
model.fit(datagen.flow(X_train_original, y_train_original, batch_size=32), epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the model
model.save('models/model_with_augmentation_combined.h5')
