import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from scikeras.wrappers import KerasClassifier


train_df = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\archive\\sign_mnist_train.csv")
test_df  = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\archive\\sign_mnist_test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df.head()
y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values
# Pour CNN on garde shape (n,28,28,1)
X_train_img = X_train.reshape(-1,28,28,1).astype('float32')
X_test_img  = X_test.reshape(-1,28,28,1).astype('float32')

print("Valeur manquantes X_train:", np.isnan(X_train).sum())
print("valeurs manquantes X_test :", np.isnan(X_test).sum())
print("Nombre d'images train:", X_train.shape[0])
print("Nombre d'images test :", X_test.shape[0])
unique_train = np.unique(y_train)
unique_test  = np.unique(y_test)
print("Classes uniques (train):", unique_train)
print("Nombre de classes (train):", len(unique_train))
print("Classes uniques (test):", unique_test)
print("Nombre de classes (test):", len(unique_test))

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    img = X_train_img[i].squeeze()  # reshape 28x28x1 → 28x28
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Exemples d'images du dataset d'entraînement")
plt.show()

plt.figure(figsize=(14,4))
sns.countplot(x=y_train)
plt.title("Distribution des classes (train)")
plt.xlabel("Label (numérique)")
plt.ylabel("Nombre d'images")
plt.show()

sums = X_train.sum(axis=1)
plt.figure(figsize=(8,4))
plt.boxplot(sums, vert=False)
plt.title("Boxplot des sommes des pixels par image (train)")
plt.xlabel("Somme des pixels")
plt.show()

mean_img = X_train_img.mean(axis=0).squeeze()
var_img  = X_train_img.var(axis=0).squeeze()
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Image moyenne (train)")
plt.imshow(mean_img, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Variance pixel-wise (train)")
plt.imshow(var_img, cmap='hot')
plt.axis('off')
plt.show()

X_train_img = X_train_img / 255.0
X_test_img  = X_test_img / 255.0
num_classes = len(np.unique(y_train))
y_train_cat = utils.to_categorical(y_train, num_classes+1)
y_test_cat  = utils.to_categorical(y_test, num_classes+1)
print("Shape X_train:", X_train_img.shape)
print("Shape y_train:", y_train_cat.shape)
#  Remappage des labels
unique_labels = sorted(np.unique(y_train))
mapping = {old: new for new, old in enumerate(unique_labels)}

y_train_mapped = np.array([mapping[v] for v in y_train])
y_test_mapped  = np.array([mapping[v] for v in y_test])

num_classes = len(unique_labels)
print("Nombre de classes corrigées :", num_classes)
#  One-hot encoding
y_train_cat = utils.to_categorical(y_train_mapped, num_classes=num_classes)
y_test_cat  = utils.to_categorical(y_test_mapped, num_classes=num_classes)

#  Visualisation des images
plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train_img[i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train_mapped[i]}")
    plt.axis("off")
plt.suptitle("Exemples d'images")
plt.show()
#  Définition du modèle CNN
def build_model(optimizer="adam"):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
# Grid Search CV = 4
cnn_sklearn = KerasClassifier(model=build_model, verbose=0)

param_grid = {
    'batch_size': [64, 128],
    'epochs': [5, 10],
    'optimizer': ['adam', 'rmsprop']
}

grid = GridSearchCV(
    estimator=cnn_sklearn,
    param_grid=param_grid,
    cv=4,
    n_jobs=-1
)

print("\n Recherche des meilleurs hyperparamètres...")
grid_result = grid.fit(X_train_img, y_train_cat)

print("\n Meilleurs paramètres trouvés :")
print(grid_result.best_params_)
print("Meilleure accuracy CV =", grid_result.best_score_)
#  Entraînement final avec meilleurs paramètres
best = grid_result.best_params_
model = build_model(optimizer=best["optimizer"])

history = model.fit(
    X_train_img, y_train_cat,
    epochs=best["epochs"],
    batch_size=best["batch_size"],
    validation_split=0.2
)
#  Courbes Loss / Accuracy
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss")
plt.legend()

plt.show()
#  Évaluation finale
test_loss, test_acc = model.evaluate(X_test_img, y_test_cat)
print("\n Accuracy sur test =", test_acc)
#  Matrice de confusion

y_pred = model.predict(X_test_img)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_mapped, y_pred_labels)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion – CNN")
plt.xlabel("Labels prédits")
plt.ylabel("Labels réels")
plt.show()