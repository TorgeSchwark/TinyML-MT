import os

def check_yolo_integrity(dataset_dir='./huggingface/mvtec_annotated'):
    splits = ['train', 'val', 'test']
    image_exts = ('.jpg', '.jpeg', '.png')

    print("🔍 Überprüfe YOLO-Dataset-Integrität:\n")

    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⚠️  Split '{split}' fehlt entweder bei den Bildern oder den Labels.")
            continue

        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(image_exts)}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')}

        missing_labels = images - labels
        extra_labels = labels - images
        matched = images & labels

        print(f"📁 Split: {split}")
        print(f"  🖼️  Bilder: {len(images)}")
        print(f"  🏷️  Labels: {len(labels)}")
        print(f"  ✅ Übereinstimmend: {len(matched)}")

        if missing_labels:
            print(f"  ❌ Fehlende Labels: {len(missing_labels)}")
            # Optional: Liste ausgeben:
            # for name in sorted(missing_labels): print(f"    - {name}")
        else:
            print("  ✅ Alle Bilder haben zugehörige Labeldateien.")

        if extra_labels:
            print(f"  ⚠️  Überflüssige Labels (kein passendes Bild): {len(extra_labels)}")
        else:
            print("  ✅ Keine überflüssigen Labeldateien.\n")

    print("🧹 Integritätsprüfung abgeschlossen.\n")
