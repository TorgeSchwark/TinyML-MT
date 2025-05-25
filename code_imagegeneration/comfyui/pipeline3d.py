import subprocess
import os
from tqdm import tqdm
import random

PATH = "huggingface/10classes"
SCRIPT = "threestudio.sh"
CONFIG = "custom/threestudio-mvimg-gen/configs/stable-zero123.yaml"

for root, dirs, files in os.walk(PATH):
    for file in tqdm(files, desc=f"Processing files in {root}"):
        if file.lower().endswith('.png'):
            img_path = os.path.join(root, file)
            print(f"Processing image: {img_path}")
            new_seed = random.getrandbits(64)
            command = f"./threestudio.sh --config {CONFIG} --train --gpu 2 data.image_path=../{img_path} exp_root_dir=../{PATH} seed={new_seed}"

            subprocess.run(command, shell=True, executable="/bin/zsh")
