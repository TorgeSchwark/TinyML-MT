import subprocess
import os
from tqdm import tqdm
import random
import shutil

PATH = "huggingface/10classes"
SCRIPT = "threestudio.sh"
CONFIG = "custom/threestudio-mvimg-gen/configs/stable-zero123.yaml"
FOLDERS = ["apple", "coffee", "cucumber", "avocado", "banana", "fruit tea", "lemon", "oatmeal", "tomato sauce"]
REMOVE_BG = False

def move_file(folder, pos):
    mvimg_dir = os.path.join(PATH,"mvimg-gen-zero123-sai")
    if os.path.isdir(mvimg_dir):
        print("FOUND")
        subfolders = [os.path.join(mvimg_dir, d) for d in os.listdir(mvimg_dir) if os.path.isdir(os.path.join(mvimg_dir, d))]
        if subfolders:
            latest_folder = max(subfolders, key=os.path.getctime)
            it0_test_dir = os.path.join(latest_folder, "save", "it0-test")
            if os.path.isdir(it0_test_dir):
                output_dir = os.path.join(os.path.join(PATH, folder), os.path.splitext(file)[0])
                output_dir = os.path.join(output_dir, pos)
                os.makedirs(output_dir, exist_ok=True)
                for gen_file in os.listdir(it0_test_dir):
                    src = os.path.join(it0_test_dir, gen_file)
                    dst = os.path.join(output_dir, gen_file)
                    os.rename(src, dst)
        # Delete the mvimg-gen-zero123-sai folder completely
        shutil.rmtree(mvimg_dir)

for folder in FOLDERS:
    folder_path = os.path.join(PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder_path}"):
        if file.lower().endswith('.png'):
            img_path = os.path.join(folder_path, file)
            print(f"Processing image: {img_path}")

            # Execute Threestudio commands

            # From sides
            new_seed = random.getrandbits(64)
            command = f"./{SCRIPT} --config {CONFIG} --train --gpu 2 data.image_path=\"../{img_path}\" exp_root_dir=\"../{PATH}\" seed={new_seed}"
            subprocess.run(command, shell=True, executable="/bin/zsh")
            move_file(folder, "sides")

            # From top
            new_seed = random.getrandbits(64)
            command = f"./{SCRIPT} --config {CONFIG} --train --gpu 2 data.image_path=\"../{img_path}\" exp_root_dir=\"../{PATH}\" seed={new_seed} data.random_camera.eval_elevation_deg=-30 data.random_camera.n_test_views=4"
            subprocess.run(command, shell=True, executable="/bin/zsh")
            move_file(folder, "bottom")

            # From top
            new_seed = random.getrandbits(64)
            command = f"./{SCRIPT} --config {CONFIG} --train --gpu 2 data.image_path=\"../{img_path}\" exp_root_dir=\"../{PATH}\" seed={new_seed} data.random_camera.eval_elevation_deg=60 data.random_camera.n_test_views=4"
            subprocess.run(command, shell=True, executable="/bin/zsh")
            move_file(folder, "top")
                
                
