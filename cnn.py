# cnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import cv2

class CarDamageClassifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # First Convolutional Block
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(MaxPooling2D((2, 2)))

        # Second Convolutional Block
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Third Convolutional Block
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        # Fully connected layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))  # Regularization layer to reduce overfitting
        model.add(Dense(1, activation='sigmoid'))  # Binary classification: damaged or not

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_images, train_labels, validation_images, validation_labels, epochs=10, batch_size=32):
        self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_images, validation_labels))

    def save_model(self, model_path="car_damage_model.h5"):
        self.model.save(model_path)

    def load_model(self, model_path="car_damage_model.h5"):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image):
        image = cv2.resize(image, (256, 256))  # Resize image to match model input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image

        prediction = self.model.predict(image)
        return "Damaged" if prediction[0] > 0.5 else "Not Damaged"
