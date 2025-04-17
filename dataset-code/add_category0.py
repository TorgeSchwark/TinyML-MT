import os
import json

# TODO !!!!!!! i dont think this is longer Necesarry (outdated)
# Adds a category 0 since it is needed in the yolo.yaml this should be fixed since its a bad work around!!!!!
def add_test_category_to_coco(json_folder):
    """
    Fügt jeder COCO JSON-Datei im angegebenen Ordner eine Kategorie mit der ID 0 und dem Namen 'Test' hinzu.
    
    :param json_folder: Ordner, der die COCO JSON-Dateien enthält.
    """
    
    # Durchlaufe alle .json-Dateien im Ordner
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            
            try:
                # Öffne und lade die JSON-Datei
                with open(json_path, 'r') as f:
                    coco_data = json.load(f)
                
                # Überprüfe, ob die Kategorie mit ID 0 bereits vorhanden ist
                categories = coco_data.get('categories', [])
                category_ids = [category['id'] for category in categories]
                
                if 0 not in category_ids:
                    # Füge die "Test"-Kategorie mit ID 0 hinzu
                    coco_data['categories'].insert(0, {
                        "supercategory": "test", 
                        "id": 0, 
                        "name": "Test"
                    })
                    print(f"Füge Kategorie 'Test' mit ID 0 zu {filename} hinzu.")
                
                    # Speichere die aktualisierte Datei
                    with open(json_path, 'w') as f:
                        json.dump(coco_data, f, indent=4)
                else:
                    print(f"Kategorie 'Test' mit ID 0 ist bereits in {filename} vorhanden.")
            
            except Exception as e:
                print(f"Fehler beim Verarbeiten der Datei {filename}: {e}")

# Beispiel: Den Ordnerpfad zu deinen COCO JSON-Dateien anpassen
json_folder = '/data22/stu236894/GitRepos/TinyML-MT/Dataset/annotations'

add_test_category_to_coco(json_folder)
