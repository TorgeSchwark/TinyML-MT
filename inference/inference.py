import tensorflow as tf
import RPi.GPIO as GPIO
import numpy as np
from picamera2 import Picamera2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import  Compose, ToTensor
import os
#import pytorch 

# TFLite-Modell laden
interpreter = tf.lite.Interpreter(model_path="/home/torge/Desktop/TinyML-MT/training-code/quantization/good-tf_model/good-model_float32.tflite")
interpreter.allocate_tensors()

# Eingabe- und Ausgabetensoren abrufen
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details)

# Eingabe- und Ausgabequantisierungsparameter
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

def preprocess_input(input_data, input_scale, input_zero_point, type):
    """Wandelt die Eingabedaten in das quantisierte Format um."""
    input_data = np.round(input_data / input_scale + input_zero_point).astype(type)
    return input_data

def dequantize_output(output_data, output_scale, output_zero_point):
    """Wandelt die quantisierten Ausgabedaten zurück in Float32."""
    return (output_data.astype(np.float32) - output_zero_point) * output_scale

# GPIO für Button einrichten
BUTTON_GPIO = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Kamera initialisieren
picam2 = Picamera2()

# Kamera konfigurieren (Vorschau-Konfiguration verwenden)
camera_config = picam2.create_video_configuration({"size": (800, 600)})
picam2.configure(camera_config)

# Temporärer Ordner für gespeicherte Bilder
TEMP_DIR = "./temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Kamera starten
picam2.start()

try:
    for i in range(10):
        print("Waiting for button press to capture image...")
        GPIO.wait_for_edge(BUTTON_GPIO, GPIO.FALLING)

        # Bild aufnehmen
        frame = picam2.capture_array()  # Erfasst ein RGB888-Bild als NumPy-Array
        print("Image captured!")

        # Bild speichern
        temp_image_path = os.path.join(TEMP_DIR, f"captured_image_{i}.jpg")
        picam2.capture_file(temp_image_path)
        print(f"Image saved as {temp_image_path}")

        # Bild wieder laden
        image = Image.open(temp_image_path)
        # Bild anzeigen
        plt.imshow(image)
        plt.title("Captured Image (JPG)")
        plt.axis("off")
        plt.show()
        image = image.resize((200, 200))  # Modellgröße anpassen (falls nötig)

        

        # Bild normalisieren
        image_normal = np.array(image) / 255.0 # in hp.load_image_labels_classify ...
        image = (image_normal * 255).astype(np.uint8) # in ImagePriceDataset
        transform_test = Compose([
            ToTensor()                         # Konvertiere das Bild zu einem Tensor
        ])
        input_data_less_dim = transform_test(Image.fromarray(image))
        input_data_less_dim = np.transpose(input_data_less_dim, (1, 2, 0))

        # input_data = preprocess_input(np.expand_dims(image, axis=0), input_scale, input_zero_point)
        input_data = np.expand_dims(input_data_less_dim, axis=0)
        # Dimensionen von input_data anzeigen
        print("Dimensionen der Eingabedaten:", input_data.shape)

        
        # Eingabedaten setzen
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Inferenz durchführen
        interpreter.invoke()

        # Ergebnisse abrufen und dequantisieren  
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # output_data = dequantize_output(output_data, output_scale, output_zero_point)

        print(output_data)
        # Ergebnisse runden
        output_data = np.round(output_data).astype(int)
        print("Inference result (dequantized):", output_data)

finally:
    # Ressourcen aufräumen
    GPIO.cleanup()
    picam2.stop()
    print("Cleaned up resources.")
