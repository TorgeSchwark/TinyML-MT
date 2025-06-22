import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.yolo as fouy
from fiftyone import ViewField as F

# YOLO format dataset directory (with subfolders: images/, labels/, data.yaml)
dataset_dir = "huggingface/mvtec_mapped/mvtec_annotated"#"huggingface/ai_checkout/artificial_rotated_mult_back/"#"huggingface/mvtec_mapped/full_classes_trained_on_10classes/"

# Import as YOLO
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",  # or "val", "test" — required for YOLOv5
)

print(dataset.get_field_schema(flat=True))

view = dataset.match(F("ground_truth.detections.label").length() == 1)

# Launch the app
session = fo.launch_app(view=view)
#session = fo.launch_app(dataset)

print("FiftyOne app is running. Press ENTER to stop...")
input()