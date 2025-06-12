import sys
from rembg import remove
from PIL import Image
import os

def remove_background(input_path):
    # Load the image
    with open(input_path, 'rb') as f:
        input_image = f.read()

    # Remove background
    output_image = remove(input_image)

    # Save result as transparent PNG
    output_path = os.path.splitext(input_path)[0] + "_nobg.png"
    with open(output_path, 'wb') as f:
        f.write(output_image)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_background.py path/to/image.jpg")
    else:
        remove_background(sys.argv[1])