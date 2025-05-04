#!/usr/bin/env zsh

source ComfyUI/.comfyui/bin/activate
python ComfyUI/main.py --disable-metadata --output-directory "huggingface/ai_shelf" --user-directory "code_imagegeneration/comfyui" --cuda-device 3

read -p "Press enter to exit"