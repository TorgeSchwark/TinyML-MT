import time
import os
import ast
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Initialize and configure the camera
picam = Picamera2()
config = picam.create_video_configuration({"size": (800, 600)})
picam.configure(config)

# Setup GPIO for button press
BUTTON_GPIO = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Target directory for storing images
data_to_path = "./Dataset"
id_file_path = os.path.join(data_to_path, "id.txt")
prices_file_path = os.path.join(data_to_path, "prices.txt")
subfder = "custom"
data_to_path = os.path.join(data_to_path, subfder)
prices = {}
print(data_to_path)
if not os.path.exists(data_to_path):
    os.makedirs(data_to_path)

# Initialize or read the current image ID
if not os.path.exists(id_file_path):
    with open(id_file_path, "w") as f:
        f.write("0")  # Start with ID 0

with open(id_file_path, "r") as f:
    current_id = int(f.readline().split(" ")[1])

# Load prices from prices.txt
if not os.path.exists(prices_file_path):
    print(f"Error: {prices_file_path} not found.")
    exit(1)

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

# Ergebnis ausgeben
print("Kombiniertes Dictionary:")
print(prices)

# Ask for the number of entries and images per object
num_entries = int(input("How many samples should be added? "))
num_images_per_entry = int(input("How many images per object? "))

# Valid object IDs and response keywords
valid_ids = list(prices.keys())
negative = ["n", "N", "break", "Break", "exit", "stop", "return"]
positive = ["y", "Y", "continue"]

# Function to capture images
def capture_images(objects_id_list, total_price, num_images):
    global current_id
    picam.start()
    print("Waiting for button press to capture image...")
    GPIO.wait_for_edge(BUTTON_GPIO, GPIO.FALLING)
    for j in range(num_images):
        time.sleep(0.3)

        # Save image with the current ID
        file_path = os.path.join(data_to_path, f"image_{current_id}.jpg")
        picam.capture_file(file_path)
        print(f"Image saved: {file_path}")

        # Save object ID list and total price for the current image
        metadata_file_path = os.path.join(data_to_path, f"image_{current_id}.txt")
        with open(metadata_file_path, "w") as metadata_file:
            metadata_file.write(f"Objects: {objects_id_list}\n")
            metadata_file.write(f"Total Price: {total_price}\n")

        # Increment the ID and update id.txt
        current_id += 1
        with open(id_file_path, "r") as f:
            lines = f.readlines()

        if lines:
            lines[0] = "current_id: " + str(current_id) + "\n"
        else:
            lines.append("current_id: " + str(current_id) + "\n")

        with open(id_file_path, "w") as f:
            f.writelines(lines)

    picam.stop()

# Loop for capturing images
for i in range(num_entries):
    ids = True
    objects_id_list = {}

    while ids:
        id_answer = input("Enter object IDs for the image (y to finish, n to clear IDs and start over): ")
        if id_answer in positive:
            print("The IDs will be recorded for the following images.")
            print("Selected object IDs:", objects_id_list)

            confirm = input("Confirm selection (y/n): ").lower()
            if confirm in positive:
                print("Preparing to capture images...")
                break
            else:
                print("Starting new selection...")
                objects_id_list = {}
        elif id_answer in negative:
            objects_id_list = {}
            print("IDs have been cleared. Start entering again:")
        elif id_answer.isdigit() and int(id_answer) in valid_ids:
            amount_answer = input("How many of these?: ")
            if amount_answer.isdigit():
                objects_id_list[int(id_answer)] = int(amount_answer)
            else:
                print("Invalid amount. Please try again.")
        else:
            print("Invalid response. Please try again.")

    # Calculate total price for the objects
    total_price = sum(prices[obj_id][1] * count for obj_id, count in objects_id_list.items())

    # Capture images
    capture_images(objects_id_list, total_price, num_images_per_entry)

print("Program finished.")
GPIO.cleanup()
