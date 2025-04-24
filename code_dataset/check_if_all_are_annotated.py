import os
import json

# 🔧 Pfade anpassen
image_folder = './Dataset/images/'  # Wo alle Bilder liegen
annotation_folder = './Dataset/annotations/'  # Alle COCO-JSON-Dateien

# 🧠 Alle Bild-Dateinamen aus dem Ordner laden (ohne Erweiterung)
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

# 🔍 Alle .json-Dateien im Annotationsordner
json_files = [f for f in os.listdir(annotation_folder) if f.endswith('.json')]
splits = [os.path.splitext(f)[0] for f in json_files]

# 📦 Initialisierung
annotated_images = {}
annotation_counts = {}
image_counts = {}
total_annotations = 0

# 🔁 Alle JSON-Dateien durchlaufen
for split in splits:
    path = os.path.join(annotation_folder, f'{split}.json')
    with open(path, 'r') as f:
        coco = json.load(f)

    annotations = coco.get('annotations', [])
    images = coco.get('images', [])

    annotation_counts[split] = len(annotations)
    image_counts[split] = len(images)
    total_annotations += len(annotations)

    for img in images:
        name = os.path.splitext(img['file_name'])[0]
        if name in annotated_images:
            annotated_images[name].append(split)
        else:
            annotated_images[name] = [split]

annotated_set = set(annotated_images.keys())

# 📊 Vergleiche
missing_in_annotations = image_files - annotated_set
extra_in_annotations = annotated_set - image_files
duplicates = {name: s for name, s in annotated_images.items() if len(s) > 1}

# 🖨️ Ergebnisse anzeigen
print("🔎 Ergebnis der Überprüfung:\n")

print(f"📄 Anzahl der annotierten Bilder (gesamt): {len(annotated_set)}\n")

if missing_in_annotations:
    print(f"❌ {len(missing_in_annotations)} Bilder sind NICHT in den Annotationen enthalten.")
else:
    print("✅ Alle Bilder sind annotiert.")

print("\n📊 Übersicht:\n")
print(f"🖼️  Gesamtanzahl der Bilder im Ordner: {len(image_files)}")
print(f"📝 Anzahl der annotierten Bilder: {len(annotated_set)}")

# ➕ Übersicht: Annotationen und Bildanzahl nach Split
print("\n🧮 Annotationen & Bilder pro Split:")
for split in splits:
    a_count = annotation_counts.get(split, 0)
    i_count = image_counts.get(split, 0)
    print(f"  - {split}: {i_count} Bilder, {a_count} Annotationen")

# ➕ Gesamtanzahl aller Annotationen
print(f"\n🧾 Gesamtanzahl aller Annotationen: {total_annotations}")

# ⚠️ Überschüssige Annotationen
if extra_in_annotations:
    print(f"\n⚠️ {len(extra_in_annotations)} Bilder wurden annotiert, sind aber NICHT im Bildordner vorhanden:")
    for name in sorted(extra_in_annotations):
        print(f"  - {name}")
else:
    print("✅ Keine überschüssigen Annotationen.")

# ⚠️ Duplikate zwischen Splits
if duplicates:
    print(f"\n⚠️ {len(duplicates)} Bilder sind in MEHR ALS EINEM Split enthalten:")
    for name, s in sorted(duplicates.items()):
        pass
        #print(f"  - {name}: {', '.join(s)}")
else:
    print("✅ Keine Duplikate zwischen Splits.")

print("\n🧹 Fertig geprüft.")
