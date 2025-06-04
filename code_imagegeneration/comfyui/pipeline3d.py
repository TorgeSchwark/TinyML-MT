import subprocess
import os
from tqdm import tqdm
import random
import shutil

PATH = "huggingface/testing"
SCRIPT = "threestudio.sh"
CONFIG = "custom/threestudio-mvimg-gen/configs/stable-zero123.yaml"
FOLDERS = ["testing"]
REMOVE_BG = False

for folder in FOLDERS:
    folder_path = os.path.join(PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.lower().endswith('.png'):
                img_path = os.path.join(root, file)
                print(f"Processing image: {img_path}")

                # Execute Threestudio command
                new_seed = random.getrandbits(64)
                command = f"./{SCRIPT} --config {CONFIG} --train --gpu 2 data.image_path=\"../{img_path}\" exp_root_dir=\"../{PATH}\" seed={new_seed}"
                subprocess.run(command, shell=True, executable="/bin/zsh")

                # Find the newest folder in mvimg-gen-zero123-sai
                mvimg_dir = os.path.join(PATH,"mvimg-gen-zero123-sai")
                if os.path.isdir(mvimg_dir):
                    print("FOUND")
                    subfolders = [os.path.join(mvimg_dir, d) for d in os.listdir(mvimg_dir) if os.path.isdir(os.path.join(mvimg_dir, d))]
                    if subfolders:
                        latest_folder = max(subfolders, key=os.path.getctime)
                        it0_test_dir = os.path.join(latest_folder, "save", "it0-test")
                        if os.path.isdir(it0_test_dir):
                            output_dir = os.path.join(root, os.path.splitext(file)[0])
                            os.makedirs(output_dir, exist_ok=True)
                            for gen_file in os.listdir(it0_test_dir):
                                src = os.path.join(it0_test_dir, gen_file)
                                dst = os.path.join(output_dir, gen_file)
                                os.rename(src, dst)
                    # Delete the mvimg-gen-zero123-sai folder completely
                    shutil.rmtree(mvimg_dir)
