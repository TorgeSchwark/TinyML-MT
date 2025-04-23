import os
import json
from shutil import copyfile

# Pfade anpassen!
image_src_folder = './Dataset/images/'  # Alle Bilder liegen hier
annotation_folder = './Dataset/annotationsoriginal/'  # Enthält train.json, val.json, test.json
image_output_base = './Dataset/images/'  # Zielordner für Bilder (train/, val/, test/)
annotation_output_base = './Dataset/annotations/'  # Zielordner für YOLO-Annotationsdateien (train/, val/, test/)

splits = ['train', 'val']

def load_categories(coco_data):
    cats = coco_data['categories']
    return {cat['id']: cat['id'] for cat in cats}  # COCO-ID == YOLO-ID (vorher)

def convert_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

def process_split(split_name):
    print(f"🔁 Verarbeite Split: {split_name}")

    annotation_path = os.path.join(annotation_folder, f"{split_name}.json")
    with open(annotation_path) as f:
        coco_data = json.load(f)

    image_id_to_filename = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco_data['images']}
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    cat_mapping = load_categories(coco_data)

    # Zielordner anlegen
    out_img_dir = os.path.join(image_output_base, split_name)
    out_ann_dir = os.path.join(annotation_output_base, split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    for img_id, (filename, width, height) in image_id_to_filename.items():
        annotations = image_annotations.get(img_id, [])
        yolo_lines = []

        for ann in annotations:
            orig_cat_id = ann['category_id']
            if orig_cat_id == 0:
                continue  # ID 0 überspringen

            cat_id = orig_cat_id - 1  # IDs um 1 verringern
            bbox = convert_bbox(ann['bbox'], width, height)
            line = f"{cat_id} {' '.join([f'{x:.6f}' for x in bbox])}"
            yolo_lines.append(line)

        # YOLO-Annotationstextdatei speichern
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(out_ann_dir, txt_filename), 'w') as f:
            f.write("\n".join(yolo_lines))

        # Bild kopieren
        src_img_path = os.path.join(image_src_folder, filename)
        dst_img_path = os.path.join(out_img_dir, filename)
        if not os.path.exists(dst_img_path):
            copyfile(src_img_path, dst_img_path)

    print(f"✅ {split_name} abgeschlossen: {len(image_id_to_filename)} Bilder verarbeitet.\n")

# Hauptprogramm
for split in splits:
    process_split(split)

print("🎉 Alle Splits fertig konvertiert!")
