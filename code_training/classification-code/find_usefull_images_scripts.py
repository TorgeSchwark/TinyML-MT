import os

def get_mvtec_with_classes(class_list, image_path, annotation_path, map_ids=None):
    valid_image_paths = []
    valid_labels = []  # Liste von Listen: Jede innere Liste enthält Labelzeilen für ein Bild

    for subfolder in os.listdir(annotation_path):
        folder_path = os.path.join(annotation_path, subfolder)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            converted_lines = []
            all_valid = True

            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])

                if class_id not in class_list:
                    all_valid = False
                    break

                # Mappe die ID falls map_ids gegeben ist
                new_class_id = map_ids[class_id] if map_ids and class_id in map_ids else class_id
                converted_line = f"{new_class_id} " + " ".join(parts[1:])
                converted_lines.append(converted_line)

            if all_valid:
                image_filename = os.path.splitext(file)[0] + ".jpg"
                image_full_path = os.path.join(image_path, subfolder, image_filename)

                if os.path.exists(image_full_path):
                    valid_image_paths.append(image_full_path)
                    valid_labels.append(converted_lines)

    return valid_image_paths, valid_labels


# To test on a NN that was trained on first_artificial_dataset!
def get_mvtec_images_for_first_artificial_dataset_classes():
    map_ids = {25: 0, 26: 0, 27: 0, 28: 1, 50: 4, 51: 3, 30: 2, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4}

    class_list = [25, 26, 27, 28, 50, 51, 30, 44, 45, 46, 47, 48, 49]

    image_paths, label_lines = get_mvtec_with_classes(
        class_list=class_list,
        image_path="../../huggingface/mvtec_annotated/images",
        annotation_path="../../huggingface/mvtec_annotated/labels",
        map_ids=map_ids
    )
    return image_paths, label_lines

# To test on a NN that was trained on 10_classes Dataset (just other translation)!
def get_mvtec_images_for_first_artificial_dataset_classes_trained_on_10_clases():
    map_ids = {25: 0, 26: 0, 27: 0, 28: 1, 50: 5, 51: 4, 30: 2, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5}
    class_list = [25, 26, 27, 28, 50, 51, 30, 44, 45, 46, 47, 48, 49]

    # there is no lemen oat meal or tomato souce in mvtec
    image_paths, label_lines = get_mvtec_with_classes(
        class_list=class_list,
        image_path="../../huggingface/mvtec_annotated/images",
        annotation_path="../../huggingface/mvtec_annotated/labels",
        map_ids=map_ids
    )
    return image_paths, label_lines


def get_mvtec_images_for_10classes_dataset():
    map_ids =  {25: 0, 26: 0, 27: 0, 28: 1, 50: 5, 51: 4, 30: 2, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 21: 3, 22: 3, 20:3}
    class_list = [25, 26, 27, 28, 50, 51, 29, 30, 21, 20, 22, 23, 44, 45, 46, 47, 48, 49, 50]

    # there is no lemen oat meal or tomato souce in mvtec
    image_paths, label_lines = get_mvtec_with_classes(
        class_list=class_list,
        image_path="../../huggingface/mvtec_annotated/images",
        annotation_path="../../huggingface/mvtec_annotated/labels",
        map_ids=map_ids
    )
    return image_paths, label_lines

import ast
import os
from pathlib import Path

def filter_dataset_return_lists(
    useful_classes,
    path_input
):
    path_input = Path(path_input)

    valid_image_paths = []
    label_dicts = []

    for folder in os.listdir(path_input):
        combined_path = path_input / folder

        if combined_path.is_dir():
            for txt_file in combined_path.glob("*.txt"):
                with open(txt_file, "r") as f:
                    first_line = f.readline()
                    if first_line.startswith("Objects:"):
                        dict_str = first_line.replace("Objects:", "").strip()
                        try:
                            obj_dict = ast.literal_eval(dict_str)
                            keys = obj_dict.keys()

                            if all(k in useful_classes for k in keys):
                                # Bildpfad ermitteln
                                jpg_file = txt_file.with_suffix(".jpg")
                                if jpg_file.exists():
                                    valid_image_paths.append(jpg_file)
                                    label_dicts.append(obj_dict)
                                else:
                                    # print(f"{jpg_file} fehlt ❌")
                                    pass
                            else:
                                # print(f"{txt_file}: ❌ enthält unerwünschte Klassen")
                                pass 
                        except Exception as e:
                            # print(f"{txt_file}: Fehler beim Parsen - {e}")
                            pass 


    return valid_image_paths, label_dicts


def get_custom_small_class_dataset():
    useful_classes_small = [1, 2, 3, 4, 5, 48, 26]
    path_input = "../../Dataset/local_dataset_all"
    
    image_paths, label_data = filter_dataset_return_lists(
        useful_classes=useful_classes_small,
        path_input=path_input
    )
    
    return image_paths, label_data

def get_custom_10class_class_dataset():
    useful_classes_small = [1, 2, 3, 4, 5, 48, 26, 13, 9]
    path_input = "../../Dataset/local_dataset_all"
    
    image_paths, label_data = filter_dataset_return_lists(
        useful_classes=useful_classes_small,
        path_input=path_input
    )
    
    return image_paths, label_data

