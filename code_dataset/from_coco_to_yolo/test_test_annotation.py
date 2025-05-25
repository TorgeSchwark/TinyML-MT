import json

def count_annotated_images(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotated_image_ids = {ann["image_id"] for ann in data["annotations"]}
    num_annotated_images = len(annotated_image_ids)

    print(f"Anzahl annotierter Bilder: {num_annotated_images}")
    return num_annotated_images


count_annotated_images("path/to/annotations.json")