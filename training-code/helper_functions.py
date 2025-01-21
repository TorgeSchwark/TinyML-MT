import random
from PIL import Image, ImageEnhance
import numpy as np 
import os 
import ast

def custom_augmentation(image):
    
    noise_factor = random.uniform(0.01, 0.03)
    image = add_random_noise(image, noise_factor)
    
    return image

def add_random_noise(image, noise_factor=0.1):
    """
    Fügt zufälliges Rauschen zu einem Bild hinzu.
    """
    np_image = np.array(image)  # Konvertiere das Bild zu einem numpy-Array
    noise = np.random.randn(*np_image.shape) * noise_factor  # Erzeuge Rauschen
    noisy_image = np_image + noise * 255  # Skaliere das Rauschen zu den Bildwerten
    noisy_image = np.clip(noisy_image, 0, 255)  # Begrenze den Wertebereich
    return Image.fromarray(np.uint8(noisy_image))  # Konvertiere zurück zu einem PIL-Image


# Custom Dataset
def load_image_label(file_name, dataset_path, image_size):
    """Helper function to load image and label from a given file."""
    # Load Image
    image_path = os.path.join(dataset_path, file_name)
    image = Image.open(image_path).resize(image_size)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    # Load Label
    txt_file = file_name.replace(".jpg", ".txt")
    txt_path = os.path.join(dataset_path, txt_file)
    label = None
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                try:
                    label = float(lines[1].split()[-1])
                except ValueError:
                    print(f"Fehler beim Lesen des Preises in Datei: {txt_path}")
    if label is None:
        raise ValueError(f"Label konnte nicht geladen werden: {txt_file}")
    return image, label


def load_image_labels_classify(file_name, dataset_path,image_size,  num_classes):
    """Helper function to load image and label (class counts) from a given file."""
    # Load Image
    image_path = os.path.join(dataset_path, file_name)
    image = Image.open(image_path).resize(image_size)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    
    # Load Label
    txt_file = file_name.replace(".jpg", ".txt")
    txt_path = os.path.join(dataset_path, txt_file)
    label = np.zeros(num_classes, dtype=np.float32)  # Initialize label vector with zeros
    
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Suche nach "Objects" und "Total Price" in der Datei
                if line.startswith("Objects:"):
                    # Extrahiere Klassen und Mengen
                    objects_info = line.split("Objects:")[1].strip()  # z.B. "{57: 1, 49: 1}"
                    objects_info = objects_info.strip("{}").split(",")
                    
                    for obj in objects_info:
                        class_id, count = map(int, obj.split(":"))
                        class_id -= 1
                        label[class_id] = count  # Füge die Anzahl der Objekte für diese Klasse hinzu
                if line.startswith("Total Price:"):
                    # Wenn der Preis benötigt wird, kann er hier extrahiert werden
                    total_price = float(line.split("Total Price:")[1].strip())
                    # Du kannst den Preis hier speichern, falls du ihn als zusätzliches Feature benötigst

    else:
        raise ValueError(f"Label-Datei nicht gefunden: {txt_file}")
    
    return image, label

def get_num_classes(prices_file_path):
    prices = {}
    with open(prices_file_path, "r") as f:
        for line in f:
            # Extrahiere das Dictionary aus der Zeile
            if ": " in line:
                _, dict_str = line.strip().split(": ", 1)
                try:
                    current_dict = ast.literal_eval(dict_str)
                    for key, value in current_dict.items():
                        if key in prices:
                            # Überprüfen, ob Werte übereinstimmen
                            if prices[key] != value:
                                print(f"Warnung: Konflikt für Schlüssel {key}: "
                                    f"{prices[key]} vs {value}")
                        else:
                            # Schlüssel hinzufügen
                            prices[key] = value
                except Exception as e:
                    print(f"Fehler beim Verarbeiten der Zeile: {line}\n{e}")

        # Finde den Schlüssel mit der höchsten ID (größte ID)
    max_id = max(prices.keys())
    return max_id

def get_combined_dict(prices_file_path):
    prices = {}
    with open(prices_file_path, "r") as f:
        for line in f:
            # Extrahiere das Dictionary aus der Zeile
            if ": " in line:
                _, dict_str = line.strip().split(": ", 1)
                try:
                    current_dict = ast.literal_eval(dict_str)
                    for key, value in current_dict.items():
                        if key in prices:
                            # Überprüfen, ob Werte übereinstimmen
                            if prices[key] != value:
                                print(f"Warnung: Konflikt für Schlüssel {key}: "
                                    f"{prices[key]} vs {value}")
                        else:
                            # Schlüssel hinzufügen
                            prices[key] = value
                except Exception as e:
                    print(f"Fehler beim Verarbeiten der Zeile: {line}\n{e}")

        # Finde den Schlüssel mit der höchsten ID (größte ID)
    return prices
