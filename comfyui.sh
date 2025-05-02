#!/usr/bin/env zsh

source ComfyUI/.comfyui/bin/activate
python ComfyUI/main.py --disable-metadata --user-directory "/data22/stu235269/TinyML-MT/code_imagegeneration/comfyui" --cuda-device 3

read -p "Press enter to exit"