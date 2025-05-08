import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from cnn import CarDamageClassifier

# 1. Set your dataset path
DATA_DIR = "Damage_car"  # Folder should contain 'damaged/' and 'not_damaged/' subfolders

def load_data(data_dir):
    images = []
    labels = []

    for label, category in enumerate(["not_damaged_car_images", "damaged_car_images"]):  # Binary classification
        folder_path = os.path.join(data_dir, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                images.append(img)
                labels.append(label)

    images = np.array(images) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels

# 2. Load the data
X, y = load_data(DATA_DIR)

# 3. Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
classifier = CarDamageClassifier()
classifier.train(X_train, y_train, X_val, y_val, epochs=10)

# 5. Save the trained model
classifier.save_model("car_damage_model.h5")
print("Model trained and saved as car_damage_model.h5")
