import os
import json
import random
from shutil import copyfile

# 🗂️ Pfade
image_src_folder = './Dataset/images/'
annotation_folder = './Dataset/annotations/'
image_output_base = './Dataset/images/'
annotation_output_base = './Dataset/labels/'

# ⚙️ Konfiguration: input_json → entweder [ratio, split] oder Liste davon
# TODO augmented images should not be in val set
split_config = {
    "D2S_training": [1.0, "train"],
    "D2S_validation": [1.0, "val"],
    "D2S_test_info": [1.0, "test"],
    "D2S_augmented": [[0.3, "val"], [0.7, "train"]],
}

def load_categories(coco_data):
    return {cat['id']: cat['id'] for cat in coco_data.get('categories', [])}

def convert_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

def split_items(items, splits):
    total = len(items)
    random.shuffle(items)
    result = {}
    start = 0
    for ratio, name in splits:
        count = int(total * ratio)
        result[name] = items[start:start + count]
        start += count
    return result

def process_config_entry(json_name, split_entries):
    annotation_path = os.path.join(annotation_folder, f"{json_name}.json")
    if not os.path.exists(annotation_path):
        print(f"⚠️ Datei nicht gefunden: {annotation_path}")
        return

    print(f"🔁 Verarbeite: {json_name}")

    with open(annotation_path) as f:
        coco_data = json.load(f)

    all_images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    image_id_to_annotations = {}
    for ann in annotations:
        image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    cat_mapping = load_categories(coco_data)

    # Normalisiere split_entries zu Liste
    if isinstance(split_entries[0], list) or isinstance(split_entries[0], tuple):
        split_instructions = split_entries
    else:
        split_instructions = [split_entries]

    # Verteile Bilder nach Split-Ratios
    split_image_sets = split_items(all_images, split_instructions)

    for split_name, images in split_image_sets.items():
        out_img_dir = os.path.join(image_output_base, split_name)
        out_ann_dir = os.path.join(annotation_output_base, split_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_ann_dir, exist_ok=True)  # Ordner wird immer angelegt

        for img in images:
            img_id = img['id']
            filename, width, height = img['file_name'], img['width'], img['height']
            anns = image_id_to_annotations.get(img_id, [])
            yolo_lines = []

            for ann in anns:
                if ann['category_id'] == 0:
                    continue
                cat_id = ann['category_id'] - 1
                bbox = convert_bbox(ann['bbox'], width, height)
                yolo_lines.append(f"{cat_id} {' '.join([f'{x:.6f}' for x in bbox])}")

            # ✏️ Nur schreiben, wenn Annotationen existieren
            if yolo_lines:
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(out_ann_dir, txt_filename), 'w') as f:
                    f.write("\n".join(yolo_lines))

            # Bild kopieren
            src_img_path = os.path.join(image_src_folder, filename)
            dst_img_path = os.path.join(out_img_dir, filename)
            if not os.path.exists(dst_img_path):
                try:
                    copyfile(src_img_path, dst_img_path)
                except FileNotFoundError:
                    print(f"❌ Bild fehlt: {filename}")

        print(f"✅ {json_name} → {split_name}: {len(images)} Bilder verarbeitet.")

# 🔁 Hauptdurchlauf
def coco_to_yolo_for_MVTEC():
    for json_name, split_entries in split_config.items():
        process_config_entry(json_name, split_entries)

        print("🎉 Alle Konvertierungen abgeschlossen.")
