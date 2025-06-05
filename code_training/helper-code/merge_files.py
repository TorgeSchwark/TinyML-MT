import shutil
from pathlib import Path

# Pfade zu den Quellverzeichnissen
source_dir1 = Path("/data22/stu236894/GitRepos/TinyML-MT/huggingface/mvtec_annotated/labels/augmented")
source_dir2 = Path("/data22/stu236894/GitRepos/TinyML-MT/huggingface/mvtec_annotated/labels/train")

# Zielverzeichnis
target_dir = Path("/data22/stu236894/GitRepos/TinyML-MT/huggingface/mvtec_annotated/labels/augmented_train")
target_dir.mkdir(parents=True, exist_ok=True)

# Alle Dateien aus dir1 kopieren
for file_path in source_dir1.iterdir():
    if file_path.is_file():
        shutil.copy(file_path, target_dir / file_path.name)

# Alle Dateien aus dir2 kopieren
for file_path in source_dir2.iterdir():
    if file_path.is_file():
        target_file = target_dir / file_path.name

        # Optional: Umbenennen, wenn Name schon existiert
        if target_file.exists():
            target_file = target_dir / f"{file_path.stem}_from_dir2{file_path.suffix}"

        shutil.copy(file_path, target_file)

print(f"Fertig: Dateien aus {source_dir1} und {source_dir2} wurden nach {target_dir} kopiert.")
