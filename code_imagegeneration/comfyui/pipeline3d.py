import subprocess
import os
from tqdm import tqdm
import random
from rembg import remove
from PIL import Image
import io

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

                with open(img_path, 'rb') as i:
                    input_data = i.read()

                img = Image.open(io.BytesIO(input_data))
                # If image has a background (not fully opaque RGBA), remove background and save as new file with _rgba
                used_img_path = img_path
                if REMOVE_BG:
                    if not (img.mode == 'RGBA' and all(a == 255 for a in img.getchannel('A').getdata())):
                        print(f"Removing background from {img_path}")
                        output_data = remove(input_data)
                        base, ext = os.path.splitext(img_path)
                        new_img_path = f"{base}_rgba{ext}"
                        with open(new_img_path, 'wb') as o:
                            o.write(output_data)
                        used_img_path = new_img_path

                new_seed = random.getrandbits(64)
                command = f"./threestudio.sh --config {CONFIG} --train --gpu 2 data.image_path=\"../{used_img_path}\" exp_root_dir=\"../{PATH}\" seed={new_seed}"
                output_dir = os.path.join(root, os.path.splitext(file)[0])
                os.makedirs(output_dir, exist_ok=True)
                generated_dir = os.path.join("mvimg-gen-zero123-sai", f"{os.path.splitext(file)[0]}_rgba_rgba.png@20250604-184721", "save", "it0-test")
                if os.path.isdir(generated_dir):
                    for gen_file in os.listdir(generated_dir):
                        src = os.path.join(generated_dir, gen_file)
                        dst = os.path.join(output_dir, gen_file)
                        os.rename(src, dst)
                subprocess.run(command, shell=True, executable="/bin/zsh")
