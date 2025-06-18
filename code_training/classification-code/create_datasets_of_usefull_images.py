import os
import shutil

def get_mvtec_with_classes(class_list, image_path, annotation_path, map_ids=None, output_path=None):
    valid_image_paths = []
    valid_labels = []

    if output_path:
        os.makedirs(os.path.join(output_path, "images/test"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels/test"), exist_ok=True)

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

                new_class_id = map_ids[class_id] if map_ids and class_id in map_ids else class_id
                converted_line = f"{new_class_id} " + " ".join(parts[1:])
                converted_lines.append(converted_line)

            if all_valid:
                image_filename = os.path.splitext(file)[0] + ".jpg"
                image_full_path = os.path.join(image_path, subfolder, image_filename)

                if os.path.exists(image_full_path):
                    valid_image_paths.append(image_full_path)
                    valid_labels.append(converted_lines)

                    if output_path:
                        # Zielpfade
                        out_image_path = os.path.join(output_path, "images/test", image_filename)
                        out_label_path = os.path.join(output_path, "labels/test", os.path.splitext(file)[0] + ".txt")

                        # Bild kopieren
                        shutil.copy2(image_full_path, out_image_path)

                        # Label-Datei schreiben
                        with open(out_label_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(converted_lines))

    return valid_image_paths, valid_labels


# To test on a NN that was trained on first_artificial_dataset!
def get_mvtec_images_for_first_artificial_dataset_classes():
    map_ids = {25: 0, 26: 0, 27: 0, 28: 1, 50: 4, 51: 3, 30: 2, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4}

    class_list = [25, 26, 27, 28, 50, 51, 30, 44, 45, 46, 47, 48, 49]

    image_paths, label_lines = get_mvtec_with_classes(
        class_list=class_list,
        image_path="../../huggingface/mvtec_annotated/images",
        annotation_path="../../huggingface/mvtec_annotated/labels",
        map_ids=map_ids,
        output_path="../../huggingface/small_class_amount_trained_on_small/"
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
        map_ids=map_ids,
        output_path="../../huggingface/small_class_amount_trained_on_10classes/"
    )
    return image_paths, label_lines


def get_mvtec_images_for_10classes_dataset():
    # alle gewollten klassen müssen übersetzt werden!!! 
    map_ids =  {25: 0, 26: 0, 27: 0, 28: 1, 50: 5, 51: 4, 30: 2, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 21: 3, 22: 3, 20:3, 23:3, 39: 8, 40: 8, 41: 8}
    class_list = [25, 26, 27, 28, 50, 51, 30, 21, 20, 22, 23, 44, 45, 46, 47, 48, 49, 39, 40, 41]

    # there is no lemen oat meal or tomato souce in mvtec
    image_paths, label_lines = get_mvtec_with_classes(
        class_list=class_list,
        image_path="../../huggingface/mvtec_annotated/images",
        annotation_path="../../huggingface/mvtec_annotated/labels",
        map_ids=map_ids,
        output_path="../../huggingface/full_classes_trained_on_10classes/"
    )
    return image_paths, label_lines

# get_mvtec_images_for_first_artificial_dataset_classes()

# get_mvtec_images_for_first_artificial_dataset_classes_trained_on_10_clases()

get_mvtec_images_for_10classes_dataset()