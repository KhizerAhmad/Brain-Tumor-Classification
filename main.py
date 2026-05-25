import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

dataset_path = "dataset/Training"
categories = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]

data = []
labels = []
image_size = 100

print("Loading Images...")

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for image_name in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data.append(image)
            labels.append(label)
        except:
            pass

print("Images Loaded Successfully")

data = np.array(data)
labels = np.array(labels)

data, labels = shuffle(data, labels, random_state=42)

print(f"Dataset Shape: {data.shape}")

unique, counts = np.unique(labels, return_counts=True)
for i, count in zip(unique, counts):
    print(f"{categories[i]}: {count}")

print("Normalizing...")
data = data / 255.0

data_flat = data.reshape(len(data), -1)

X_train, X_test, y_train, y_test = train_test_split(
    data_flat,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

print("Training models...")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

print("\n=== RESULTS ===")

models = {
    "Logistic Regression": pred_lr,
    "Decision Tree": pred_dt,
    "Random Forest": pred_rf,
    "SVM": pred_svm
}

for name, pred in models.items():
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average='weighted'))
    print("Recall:", recall_score(y_test, pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, pred, average='weighted'))

cm_lr = confusion_matrix(y_test, pred_lr)
cm_dt = confusion_matrix(y_test, pred_dt)
cm_rf = confusion_matrix(y_test, pred_rf)
cm_svm = confusion_matrix(y_test, pred_svm)

algorithms = ["Logistic Regression","Decision Tree","Random Forest","SVM"]
accuracies = [
    accuracy_score(y_test,pred_lr),
    accuracy_score(y_test,pred_dt),
    accuracy_score(y_test,pred_rf),
    accuracy_score(y_test,pred_svm)
]

plt.figure(figsize=(8,5))
plt.bar(categories, counts)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(algorithms, accuracies)
plt.title("Accuracy Comparison")
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(10,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(data[i].reshape(image_size,image_size), cmap='gray')
    plt.title(categories[labels[i]])
    plt.axis("off")
plt.tight_layout()
plt.show()

def plot_cm(cm, title):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()

    for i in range(len(categories)):
        for j in range(len(categories)):
            plt.text(j, i, cm[i, j],
                     ha='center',
                     va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

plot_cm(cm_lr, "Logistic Regression Confusion Matrix")
plot_cm(cm_dt, "Decision Tree Confusion Matrix")
plot_cm(cm_rf, "Random Forest Confusion Matrix")
plot_cm(cm_svm, "SVM Confusion Matrix")

print("\n=== SUMMARY STATISTICS ===")
df = pd.DataFrame(data_flat)
print(df.describe())