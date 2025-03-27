import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Define dataset path
dataset_path = "D:/april internship/Task 3/dogvscat/train"
categories = ["0", "1"]  # '0' for cats, '1' for dogs

# Image preprocessing parameters
img_size = 64  # Resize images to 64x64

data = []
labels = []

# Load and preprocess images
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Assign 0 to 'cats', 1 to 'dogs'
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.resize(image, (img_size, img_size))  # Resize image
            image = image.flatten()  # Flatten image into a 1D array
            data.append(image)
            labels.append(label)

# Convert to NumPy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Take user input image for classification
while True:
    image_path = input("Enter the path of the image to classify: ").strip()
    image_path = image_path.replace("\\", "/")  # Convert backslashes to forward slashes
    
    if not os.path.exists(image_path):
        print("Error: File does not exist. Please enter a valid file path.")
        continue
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.resize(image, (img_size, img_size))
        image = image.flatten().reshape(1, -1)  # Flatten and reshape for model
        prediction = svm_model.predict(image)[0]
        print("Predicted Class:", "Cat" if prediction == 0 else "Dog")
        break
    else:
        print("Error: Unable to load image. Check the file format.")
