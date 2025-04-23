import random
from PIL import Image
import numpy as np 
import os 
import ast
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

# TODO Use the helper function from code_training/helper-code/helper_functions.py
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

def load_generated_images(file_name, dataset_path, image_size, num_classes):
    """Helper function to load generated image and its label (class counts) from a given file."""
    # Lade das Bild (angenommen, das Bild hat den Namen "generated_image_x.png")
    image_path = os.path.join(dataset_path, file_name)
    try:
        image = Image.open(image_path).convert('RGB').resize(image_size)  # Konvertiere es sicher in RGB
        image = np.array(image, dtype=np.float32) / 255.0  # Normalisiere das Bild
    except Exception as e:
        raise ValueError(f"Fehler beim Laden des Bildes: {image_path}. Fehler: {e}")
    
    # Lade die zugehörige Label-Datei (angenommen, das Label hat den Namen "label_image_x.txt")
    label_file_name = file_name.replace("generated_image", "label_image").replace(".png", ".txt")
    txt_path = os.path.join(dataset_path, label_file_name)
    label = np.zeros(num_classes, dtype=np.float32)  # Initialisiere das Label-Array mit Nullen
    
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r") as f:
                # Angenommen, die Datei hat das Format [3, 1, 0] als Text
                label_line = f.readline().strip()
                label = np.array(eval(label_line), dtype=np.float32)  # Konvertiere die Liste von String zu Array
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der Label-Datei: {txt_path}. Fehler: {e}")
    else:
        raise ValueError(f"Label-Datei nicht gefunden: {label_file_name}")
    
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

class TrainModel(pl.LightningModule):
    def __init__(self, model, prices, learning_rate=1e-3, optimizer=None):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.model = model
        self.prices = prices
        self.criterion = nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.optimizer = optimizer
        self.threshold = 0.5
        self.save_hyperparameters() #ignore=['model']

    def forward(self, x):
        return self.model(x)

    def calculate_total_price(self, predictions, labels):
        """
        Berechnet den Gesamtpreis basierend auf den Vorhersagen und tatsächlichen Labels
        """
        # Vorhersagen und tatsächliche Labels basierend auf Threshold filtern
        predicted_labels = [i for i, pred in enumerate(predictions) if pred > self.threshold]
        true_labels = [i for i, label in enumerate(labels) if label > 0.5]

        # Berechnung der Gesamtpreise
        predicted_price = sum(self.prices[label][1] for label in predicted_labels if label in self.prices)
        true_price = sum(self.prices[label][1] for label in true_labels if label in self.prices)

        return predicted_price, true_price

    def step(self, batch, stage):
        x, y = batch
        y_pred = self(x)  # Vorhersagen

        # Hauptverlust berechnen (MSE auf allen Labels)
        loss = self.criterion(y_pred, y)

        # MAE und MSE der Hauptlabels
        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)

        # Berechnung der Gesamtpreise
        batch_predicted_prices = []
        batch_true_prices = []
        for i in range(len(y)):
            predicted_price, true_price = self.calculate_total_price(y_pred[i], y[i])
            batch_predicted_prices.append(predicted_price)
            batch_true_prices.append(true_price)

        # Manuelle Berechnung von MAE und MSE für Gesamtpreise
        total_mae = sum(abs(tp - pp) for tp, pp in zip(batch_true_prices, batch_predicted_prices)) / len(batch_true_prices)
        total_mse = sum((tp - pp) ** 2 for tp, pp in zip(batch_true_prices, batch_predicted_prices)) / len(batch_true_prices)

        # Logs für Progress-Bar und Training/Validierung/Test
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_mse", mse, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_total_mae", total_mae, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_total_mse", total_mse, prog_bar=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Scheduler erstellen
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.965)
        return [self.optimizer], [scheduler]


