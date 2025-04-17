import os
import json

# 🔧 Pfade anpassen
image_folder = './Dataset/images/'  # Wo alle Bilder liegen
annotation_folder = './Dataset/annotationsoriginal/'  # train.json, val.json, test.json
splits = ['train', 'val', 'test']

# 🧠 Alle Bild-Dateinamen aus dem Ordner laden (ohne Erweiterung)
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

# 👀 Alle annotierten Bilder sammeln
annotated_images = {}
for split in splits:
    path = os.path.join(annotation_folder, f'{split}.json')
    with open(path, 'r') as f:
        coco = json.load(f)
    
    for img in coco['images']:
        name = os.path.splitext(img['file_name'])[0]
        if name in annotated_images:
            annotated_images[name].append(split)
        else:
            annotated_images[name] = [split]

annotated_set = set(annotated_images.keys())

# 📊 Vergleiche
missing_in_annotations = image_files - annotated_set
extra_in_annotations = annotated_set - image_files
duplicates = {name: splits for name, splits in annotated_images.items() if len(splits) > 1}

# 🖨️ Ergebnisse anzeigen
print("🔎 Ergebnis der Überprüfung:\n")

print(f"📄 Anzahl der annotierten Bilder: {len(annotated_set)}\n")

if missing_in_annotations:
    print(f"❌ {len(missing_in_annotations)} Bilder sind NICHT in den Annotationen enthalten:")
    for name in sorted(missing_in_annotations):
        pass
else:
    print("✅ Alle Bilder sind annotiert.")
    
print("📊 Übersicht:\n")
print(f"🖼️  Gesamtanzahl der Bilder im Ordner: {len(image_files)}")
print(f"📝 Anzahl der annotierten Bilder: {len(annotated_set)}\n")


if extra_in_annotations:
    print(f"\n⚠️ {len(extra_in_annotations)} Bilder wurden annotiert, sind aber NICHT im Bildordner vorhanden:")
    for name in sorted(extra_in_annotations):
        print(f"  - {name}")
else:
    print("✅ Keine überschüssigen Annotationen.")

if duplicates:
    print(f"\n⚠️ {len(duplicates)} Bilder sind in MEHR ALS EINEM Split enthalten:")
    for name, s in sorted(duplicates.items()):
        print(f"  - {name}: {', '.join(s)}")
else:
    print("✅ Keine Duplikate zwischen Splits.")

print("\n🧹 Fertig geprüft.")
