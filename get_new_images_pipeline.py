import time
import os
from picamera2 import Picamera2, Preview

# Initialize and configure the camera
picam = Picamera2()

# Create a still configuration with high resolution
config = picam.create_preview_configuration()
# Set shutter speed (example: 10000 microseconds or 10ms)
# picam.set_controls({"ExposureTime": 4000})  # Exposure time in microseconds (10ms)

picam.configure(config)

# Target directory for storing images
data_to_path = "./Dataset"
if not os.path.exists(data_to_path):
    os.makedirs(data_to_path)

# Ask for the number of entries and images per object
num_entries = int(input("How many samples should be added? "))
num_images_per_entry = int(input("How many images per object? "))

# Start the camera preview

# Valid object IDs and response keywords
valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
negative = ["n", "N", "break", "Break", "exit", "stop", "return"]
positive = ["y", "Y", "continue"]

# Loop for capturing images
for i in range(num_entries):
    continue_accept = input("Do you want to continue? (y/n): ").lower()
    if continue_accept == "n":
        break

    ids = True
    objects_id_list = []
    id_answer = ""

    while ids:
        id_answer = input("Enter object IDs for the image (y to finish, n to clear IDs and start over): ")

        if id_answer.isdigit() and int(id_answer) in valid_ids:
            objects_id_list.append(int(id_answer))
        elif id_answer in negative:
            objects_id_list = []
            print("IDs have been cleared. Start entering again:")
        elif id_answer in positive:
            print("The IDs will be recorded for the following images.")
            print("Selected object IDs:", ", ".join(map(str, objects_id_list)))

            confirm = input("Confirm selection (y/n): ").lower()
            if confirm in positive:
                print("Preparing to capture images...")
                break
            else:
                print("Starting new selection...")
                objects_id_list = []
        else:
            print("Invalid response. Please try again.")

    # Start the camera for capturing images
    picam.start()
    for j in range(num_images_per_entry):
        time.sleep(1)
        file_path = os.path.join(data_to_path, f"sample-{i}_image-{j}.jpg")
        picam.capture_file(file_path)
        print(f"Image {j + 1} for sample {i + 1} saved: {file_path}")

    picam.stop()  # Stop the camera after capturing all images for the current entry

# Stop the camera preview and close
picam.stop_preview()
picam.close()
print("Program finished.")
