import tensorflow as tf
import RPi.GPIO as GPIO
import numpy as np
from picamera2 import Picamera2
from PIL import Image
import matplotlib.pyplot as plt

# TFLite-Modell laden
interpreter = tf.lite.Interpreter(model_path="/home/torge/Desktop/TinyML-MT/training-code/quantization/good-model_quantized.tflite")
interpreter.allocate_tensors()

# Eingabe- und Ausgabetensoren abrufen
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Eingabe- und Ausgabequantisierungsparameter
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

def preprocess_input(input_data, input_scale, input_zero_point):
    """Wandelt die Eingabedaten in das quantisierte Format um."""
    input_data = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
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

# Kamera starten
picam2.start()
for i in range(10):
    print("Waiting for button press to capture image...")
    GPIO.wait_for_edge(BUTTON_GPIO, GPIO.FALLING)

    # Bild aufnehmen
    frame = picam2.capture_array()  # Erfasst ein RGB888-Bild als NumPy-Array
    print("Image captured!")

    # Bild anzeigen
    

    # Bild vorbereiten
    image = Image.fromarray(frame)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((200, 200))  # Modellgröße anpassen (falls nötig)
    plt.imshow(frame)
    plt.title("Captured Image")
    plt.axis("off")
    plt.show()
    image = np.array(image) / 255.0  # Normalisieren auf [0, 1]
    # Quantisierung der Eingabedaten
    input_data = preprocess_input(np.expand_dims(image, axis=0), input_scale, input_zero_point)
    print(input_data)
    # Dimensionen von input_data anzeigen
    print("Dimensionen der Eingabedaten:", input_data.shape)

    # Eingabedaten setzen
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Inferenz durchführen
    interpreter.invoke()

    # Ergebnisse abrufen und dequantisieren  
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = dequantize_output(output_data, output_scale, output_zero_point)
    print(output_data)
    # Ergebnisse runden
    output_data = np.round(output_data).astype(int)
    print("Inference result (dequantized):", output_data)


# Ressourcen aufräumen
GPIO.cleanup()
picam2.stop()
