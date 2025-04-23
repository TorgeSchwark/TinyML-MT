import json
import yaml


# creates yaml directly from annotation files
import json
import yaml

# creates yaml directly from annotation files
def generate_yaml_from_coco(coco_json_path, output_yaml_path):
    """
    Extrahiert die Klassen aus einer COCO-Annotationsdatei und erstellt eine YAML-Datei.
    IDs werden um 1 reduziert (z.B. von 1 auf 0). Klassen mit ID 0 werden ignoriert.
    
    Parameters:
    - coco_json_path: str | Path – der Pfad zur COCO JSON-Annotationsdatei.
    - output_yaml_path: str | Path – der Pfad, an dem die YAML-Datei gespeichert wird.
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

    # Dataset-Pfade (können angepasst werden)
    dataset_paths = {
        'train': './Dataset/images/train',
        'val': './Dataset/images/val',
        'test': './Dataset/images/test'
    }

    # Erstellen des YAML-Inhalts
    yaml_content = {
        'train': dataset_paths['train'],
        'val': dataset_paths['val'],
        'test': dataset_paths['test'],
        'nc': len(class_names),  # Anzahl der Klassen
        'names': [class_names[i] for i in sorted(class_names.keys())]
    }

    # YAML-Datei speichern
    with open(output_yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        print(f"✅ YAML-Datei erfolgreich gespeichert: {output_yaml_path}")

# Beispielnutzung:
if __name__ == "__main__":
    coco_json_path = "./Dataset/annotations/D2S_training.json"
    output_yaml_path = 'coco11.yaml'
    
    generate_yaml_from_coco(coco_json_path, output_yaml_path)
