import os
import argparse
import subprocess
from datetime import datetime
import traceback

import wandb
from ultralytics import YOLO
from pipeline_utils import log_class_metrics_heatmap, custom_grids, custom_metrics, mvtec_grids, mvtec_metrics
import find_usefull_images_scripts as im_script


def setup_environment(gpu: str, hf_cache: str = "../../.cache"):
    # Run from owner directory
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        raise RuntimeError(f"Failed to change directory: {e}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HOME"] = hf_cache
    print(f"[ENV] Using GPU: {gpu}")
    print(f"[ENV] HF_HOME set to: {hf_cache}")

    # Set YOLO CLI settings
    subprocess.run(["yolo", "settings", "wandb=True"], check=True)

    # Verify Hugging Face login
    subprocess.run(["huggingface-cli", "whoami"], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model with W&B logging")

    parser.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES index")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=500, help="Input image size")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--name", type=str, default="yolo", help="Base name for the run")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model to use (path or name)")
    parser.add_argument("--custom", type=str, default="True", help="Evaluate custom metrics and grids")
    parser.add_argument("--mvtec", type=str, default="True", help="Use MVTec dataset settings")
    parser.add_argument("--pretrained", type=str, default="True", help="Use pretrained model weights")

    return parser.parse_args()

MVTEC_ANNOTATED = "../../huggingface/mvtec_mapped/full_classes_trained_on_10classes/dataset.yaml"
MVTEC_FUNCTION = im_script.get_mvtec_images_for_10classes_dataset
BIG = True
VAL_EVERY_EPOCH = True # Set save_period to 1 then

## -------- Callbacks --------
def val_after_epoch_callback(trainer):
    epoch = trainer.epoch
    ckpt_path = f"{trainer.save_dir}/weights/epoch{epoch}.pt"
    print(ckpt_path)
    if os.path.exists(ckpt_path):
        model = YOLO(ckpt_path)
        custom_metrics(model, BIG)
        mvtec_metrics(model, MVTEC_ANNOTATED, BIG)
    else:
        print(f"No validation possible! Checkpoint not found for epoch {epoch}")

    

def main():
    args = parse_args()
    setup_environment(args.gpu)
    args.custom = args.custom.lower() == "true"
    args.mvtec = args.mvtec.lower() == "true"
    args.pretrained = args.pretrained.lower() == "true"
    if args.custom:
        print(f"\033[1;93mTesting on MVTec with the following Settings: Path: {MVTEC_ANNOTATED}, Big: {BIG}, Function: {MVTEC_FUNCTION.__name__}\033[0m")
    dataset_path = os.path.abspath(args.dataset_path)
    # Extract model size (letter before .pt in model filename)
    model_filename = os.path.basename(args.model)
    model_size = ""
    if model_filename.endswith(".pt"):
        # Find the last letter before ".pt"
        for c in reversed(model_filename[:-3]):
            if c.isalpha():
                model_size = c
                break

    # Extract dataset name (last folder before the yaml file)
    dataset_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_dir)

    run_name = f"{args.name}_{model_size}_{dataset_name}_{datetime.now().strftime('%d%b-%H:%M:%S')}"
    if args.pretrained:
        run_name += "_pretrained"

    # Initialize W&B
    run = wandb.init(
        project="Yolo-Training",
        entity="maats",
        name=run_name,
        config={
            "model_size": model_size,
            "dataset": dataset_name,
        },
        sync_tensorboard=True,
    )

    try:
        model = YOLO(args.model)

        if VAL_EVERY_EPOCH:
            model.add_callback("on_fit_epoch_end", val_after_epoch_callback)

        results = model.train(
        data=args.dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="Yolo-Training",
        name=run_name,
        verbose=True,
        val=True,
        save=True,
        save_period=1,
        mode="wandb",
        batch=0.80, # 70% ? Check this
        patience=10, # Early Stopping Patience
        pretrained=args.pretrained, #! Pretrained Model
        multi_scale=False, #! Test this
        cos_lr=False, #! Test this
        freeze=None, #! Test this
        #augment=True, # This is for applying augmentations to prediction sources
        #hsv_h=0.1,
        #degrees=180,
        #shear=10,
        #perspective=0.0003,
        #mixup = 0.3, # das was Jannek meinte 
        #cutmix = 0.3,
        #copy_paste = 0.1, # weis ja nicht ...

        # No Augmentations:
        #hsv_h=0.0,
        #hsv_s=0.0,
        #hsv_v=0.0,
        #degrees=0.0,
        #translate=0.0,
        #scale=1.0,           # set to 1.0 to avoid scaling
        #shear=0.0,
        #perspective=0.0,
        #flipud=0.0,
        #fliplr=0.0,
        #bgr=0.0,
        #mosaic=0.0,
        #mixup=0.0,
        #cutmix=0.0,
        #copy_paste=0.0,
        #copy_paste_mode="flip",  # has no effect when copy_paste=0.0
        #auto_augment="",         # classification only
        #erasing=0.0,
    )
        wandb.init(id=run.id, resume="allow", project="Yolo-Training")
        wandb.config.update(model.args) # Log all settings to WANDB
        log_class_metrics_heatmap(results)
        # Load best model
        best_model_path = f"Yolo-Training/{run_name}/weights/best.pt"
        model = YOLO(best_model_path)

        if args.custom:
            custom_grids(model, BIG)
            custom_metrics(model, BIG) # Also handles Heatmap
        
        if args.mvtec:
            mvtec_grids(model, MVTEC_FUNCTION)
            mvtec_metrics(model, MVTEC_ANNOTATED, BIG)

    except Exception as e:
        print(f"\033[1;91mAn error occurred: {e}\033[0m")
        traceback.print_exc()

    finally:
        wandb.finish()
        print("Done")


if __name__ == "__main__":
    main()