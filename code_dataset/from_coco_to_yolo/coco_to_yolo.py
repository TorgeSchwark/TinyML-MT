from create_yaml import generate_yaml_from_coco_MVTEC
from lables_coco_to_yolo import coco_to_yolo_for_MVTEC
from check_integrety import check_yolo_integrity

def main():
    print("🚀 Starte Konvertierung von COCO nach YOLO...")
    coco_to_yolo_for_MVTEC()
    print("✅ COCO → YOLO abgeschlossen.")

    print("📄 Erstelle YAML-Datei für MVTec...")
    generate_yaml_from_coco_MVTEC()
    print("✅ YAML-Datei erstellt.")

    check_yolo_integrity()

if __name__ == "__main__":
    main()
