import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import re

# Pfad zum Dataset
DATASET_PATH = "./Dataset"
IMAGE_SIZE = (800, 600)

# Funktion, um Bilder und Labels zu laden
def load_data(dataset_path):
    images = []
    labels = []

    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)

        if file_name.endswith(".jpg"):
            # Bild verarbeiten
            image = Image.open(file_path).resize(IMAGE_SIZE)
            images.append(np.array(image) / 255.0)  # Normalisieren auf [0, 1]

            # Label aus der zugehörigen Textdatei extrahieren
            txt_file = file_name.replace(".jpg", ".txt")
            txt_path = os.path.join(dataset_path, txt_file)
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:  # Sicherstellen, dass mindestens zwei Zeilen vorhanden sind
                        try:
                            # Preis aus der zweiten Zeile extrahieren
                            price = float(lines[1].split()[-1])  # Letztes Element der zweiten Zeile
                            labels.append(price)
                        except ValueError:
                            print(f"Fehler beim Lesen des Preises in Datei: {txt_file}")
                    else:
                        print("nolines")
            else:
                print("no file named: ", txt_path)
    print(labels)
    return np.array(images), np.array(labels)


# Daten laden
images, labels = load_data(DATASET_PATH)

# Labels normalisieren (z. B. Min-Max-Scaling)

# Daten in Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Modell aufbauen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)),
    MaxPooling2D((3, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((3, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((3, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')  # Regression für kontinuierliche Werte
])

# Modell zusammenfassen
print("Model Summary:")
model.summary()

# Modell kompilieren
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Modell trainieren
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Ergebnisse auswerten
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# 20 zufällige Bilder auswählen und Vorhersagen anzeigen
num_predictions = 20
indices = np.random.choice(len(X_test), num_predictions, replace=False)
selected_images = X_test[indices]
selected_labels = y_test[indices]

predictions = model.predict(selected_images)

print("\nPredictions (Vorhersagen) und Labels (Wahre Werte):")
for i in range(num_predictions):
    print(f"Bild {i+1}: Prediction = {predictions[i][0]:.2f}, Label = {selected_labels[i]:.2f}")
    
    # Bild anzeigen
    plt.imshow(selected_images[i])
    plt.title(f"Prediction: {predictions[i][0]:.2f}, Label: {selected_labels[i]:.2f}")
    plt.axis('off')
    plt.show()