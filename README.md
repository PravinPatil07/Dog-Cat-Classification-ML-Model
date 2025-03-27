# Dog-Cat-Classification-ML-Model
here i trained the ML Model for classifying the cat and dog and predicting the image 
# Cat-Dog Classification using Support Vector Machine (SVM)


# Data Set
Dataset :- https://www.kaggle.com/c/dogs-vs-cats/data

## Dataset Information
The dataset consists of images of cats and dogs stored in the following directories:
- **Cats:** `D:/april internship/Task 3/dogvscat/train/0`
- **Dogs:** `D:/april internship/Task 3/dogvscat/train/1`

## About the Model
This project utilizes a **Support Vector Machine (SVM)** for binary image classification, distinguishing between cats and dogs. SVM is a powerful supervised learning algorithm commonly used for classification tasks. It works by finding the optimal hyperplane that best separates the two classes in a high-dimensional space.

### Why SVM?
- **Effective for binary classification** problems like distinguishing between two categories.
- **Works well on smaller datasets**, as it does not require large amounts of data for training.
- **Robust to high-dimensional data**, making it suitable for image classification.

## Steps to Implement the Model

### 1. Import Necessary Libraries
```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
```

### 2. Load and Preprocess the Dataset
```python
# Define dataset paths
cat_dir = "D:/april internship/Task 3/dogvscat/train/0"
dog_dir = "D:/april internship/Task 3/dogvscat/train/1"
img_size = 64  # Image resize dimensions

data = []
labels = []

# Load images and assign labels
for category, label in [(cat_dir, 0), (dog_dir, 1)]:
    for img_name in os.listdir(category):
        img_path = os.path.join(category, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (img_size, img_size)).flatten()
            data.append(image)
            labels.append(label)

# Convert lists to NumPy arrays
data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)
```

### 3. Split Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
```

### 4. Train the SVM Model
```python
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
```

### 5. Save the Trained Model
```python
joblib.dump(svm_model, "cat_dog_svm_model.pkl")
```

### 6. Load the Model and Make Predictions
```python
def predict_image(image_path):
    model = joblib.load("cat_dog_svm_model.pkl")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size)).flatten() / 255.0
    image = np.array([image])
    prediction = model.predict(image)
    return "Cat" if prediction[0] == 0 else "Dog"

# Example Usage
image_path = input("Enter the path of the image to classify: ")
print("Prediction:", predict_image(image_path))
```

### 7. Implement Real-time Prediction via Webcam
```python
def predict_from_camera():
    model = joblib.load("cat_dog_svm_model.pkl")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray, (img_size, img_size)).flatten() / 255.0
        image = np.array([image])
        prediction = model.predict(image)
        label = "Cat" if prediction[0] == 0 else "Dog"
        cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Cat-Dog Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

predict_from_camera()
```

## Conclusion
This project uses **Support Vector Machine (SVM)** to classify images of cats and dogs. The model is trained on grayscale images of size 64x64 pixels and can predict images from a file or a live webcam feed. The SVM algorithm is chosen for its efficiency and accuracy in binary classification tasks.

---
🚀 **Run the script and classify your pet images!** 🐶🐱
