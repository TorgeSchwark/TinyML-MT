import os
import shutil

"""
This script moves images from the test set to the training set based on the labels provided. 
This is useful for extracting some pictures from our custom MVTec test dataset and move them to training for better accuracies.
"""

# Set your base directories
base_path = "huggingface/mvtec_mapped/full_classes_with_training_samples"
labels_train_dir = os.path.join(base_path, "labels/train")
images_test_dir = os.path.join(base_path, "images/test")
images_train_dir = os.path.join(base_path, "images/train")

# Ensure destination exists
os.makedirs(images_train_dir, exist_ok=True)

# Loop through all label files that were moved
for label_file in os.listdir(labels_train_dir):
    if label_file.endswith(".txt"):
        image_name = label_file.replace(".txt", ".jpg")
        src_image_path = os.path.join(images_test_dir, image_name)
        dst_image_path = os.path.join(images_train_dir, image_name)

        if os.path.exists(src_image_path):
            shutil.move(src_image_path, dst_image_path)
            print(f"Moved: {image_name}")
        else:
            print(f"Missing image: {image_name}")