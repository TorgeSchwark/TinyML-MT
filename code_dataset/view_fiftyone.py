import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.yolo as fouy

# YOLO format dataset directory (with subfolders: images/, labels/, data.yaml)
dataset_dir = "huggingface/mvtec_yolo"

# Import as YOLO
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",  # or "val", "test" — required for YOLOv5
)

# Launch the app
session = fo.launch_app(dataset)

print("FiftyOne app is running. Press ENTER to stop...")
input()