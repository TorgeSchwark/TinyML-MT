#!/usr/bin/env zsh

source ComfyUI/.comfyui/bin/activate
export HF_HOME="./.cache"
python ComfyUI/main.py --disable-metadata --output-directory "huggingface" --user-directory "code_imagegeneration/comfyui" --cuda-device 3

read -p "Press enter to exit"