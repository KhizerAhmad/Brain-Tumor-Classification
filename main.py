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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

dataset_path = "dataset/Training"
categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

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

algorithms = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
accuracies = [
    accuracy_score(y_test, pred_lr),
    accuracy_score(y_test, pred_dt),
    accuracy_score(y_test, pred_rf),
    accuracy_score(y_test, pred_svm)
]

plt.figure(figsize=(8, 5))
plt.bar(categories, counts)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(algorithms, accuracies)
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(data[i].reshape(image_size, image_size), cmap='gray')
    plt.title(categories[labels[i]])
    plt.axis("off")
plt.tight_layout()
plt.show()

def plot_cm(cm, title):
    plt.figure(figsize=(6, 5))
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

data_rgb = []
labels_rgb = []

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)
    for image_name in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data_rgb.append(image)
            labels_rgb.append(label)
        except:
            pass

data_rgb = np.array(data_rgb) / 255.0
labels_rgb = np.array(labels_rgb)

data_rgb, labels_rgb = shuffle(data_rgb, labels_rgb, random_state=42)

labels_cat = to_categorical(labels_rgb, num_classes=4)

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    data_rgb,
    labels_cat,
    test_size=0.3,
    random_state=42,
    stratify=labels_rgb
)

print("\nTraining CNN...")

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.fit(
    X_train_dl, y_train_dl,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_dl, y_test_dl, verbose=0)
print(f"\nCNN Accuracy: {cnn_acc:.4f}")

print("\nTraining ResNet50...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size, image_size, 3)
)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

resnet_model = Model(inputs=base_model.input, outputs=output)

resnet_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

resnet_model.fit(
    X_train_dl, y_train_dl,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

resnet_loss, resnet_acc = resnet_model.evaluate(X_test_dl, y_test_dl, verbose=0)
print(f"\nResNet50 Accuracy: {resnet_acc:.4f}")

print("\n=== FINAL MODEL COMPARISON ===")
print(f"Logistic Regression : {accuracy_score(y_test, pred_lr):.4f}")
print(f"Decision Tree       : {accuracy_score(y_test, pred_dt):.4f}")
print(f"Random Forest       : {accuracy_score(y_test, pred_rf):.4f}")
print(f"SVM                 : {accuracy_score(y_test, pred_svm):.4f}")
print(f"CNN                 : {cnn_acc:.4f}")
print(f"ResNet50            : {resnet_acc:.4f}")

all_models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "CNN", "ResNet50"]
all_accuracies = [
    accuracy_score(y_test, pred_lr),
    accuracy_score(y_test, pred_dt),
    accuracy_score(y_test, pred_rf),
    accuracy_score(y_test, pred_svm),
    cnn_acc,
    resnet_acc
]

plt.figure(figsize=(10, 5))
bars = plt.bar(all_models, all_accuracies, color=['#4C72B0', '#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#DD8452'])
plt.title("All Models Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
for bar, acc in zip(bars, all_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

pred_cnn = np.argmax(cnn_model.predict(X_test_dl), axis=1)
pred_resnet = np.argmax(resnet_model.predict(X_test_dl), axis=1)
y_test_labels = np.argmax(y_test_dl, axis=1)

plot_cm(confusion_matrix(y_test_labels, pred_cnn), "CNN Confusion Matrix")
plot_cm(confusion_matrix(y_test_labels, pred_resnet), "ResNet50 Confusion Matrix")