import os
import shutil

def flatten_image_folder(source_root, target_folder, allowed_extensions=".jpg"):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    counter = 0
    for subfolder in os.listdir(source_root):
        subfolder_path = os.path.join(source_root, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if file_path.endswith(allowed_extensions):
                    shutil.copy(file_path, target_folder)
                    counter += 1
    print(f"{counter} Bilder wurden in '{target_folder}' kopiert.")

# Beispielaufruf
flatten_image_folder("./huggingface/mvtec_annotated/images", "./huggingface/mvtec_flattened")
