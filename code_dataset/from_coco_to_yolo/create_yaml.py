import json
import yaml

def generate_yaml_from_coco(coco_json_path, output_yaml_path):
    """
    Extrahiert die Klassen aus einer COCO-Annotationsdatei und erstellt eine YAML-Datei.
    IDs werden um 1 reduziert (z.B. von 1 auf 0). Klassen mit ID 0 werden ignoriert.
    """
    
    # COCO JSON-Datei laden
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Klassen extrahieren, IDs um 1 reduzieren, 0 ignorieren
    categories = coco_data['categories']
    class_names = {
        category['id'] - 1: category['name']
        for category in categories
        if category['id'] > 0
    }

    # Dataset-Pfade
    dataset_paths = {
        'train': './Dataset/images/train',
        'val': './Dataset/images/val',
    }

    # YAML-Struktur: 'names' als Dictionary mit numerischen Keys
    yaml_content = {
        'train': dataset_paths['train'],
        'val': dataset_paths['val'],
        'nc': len(class_names),
        'names': {i: class_names[i] for i in sorted(class_names)}
    }

    with open(output_yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)
        print(f"✅ YAML-Datei erfolgreich gespeichert: {output_yaml_path}")

# Beispielnutzung:
def generate_yaml_from_coco_MVTEC():
    coco_json_path = "./Dataset/annotations/D2S_training.json"
    output_yaml_path = './Dataset/coco11.yaml'
    
    generate_yaml_from_coco(coco_json_path, output_yaml_path)

