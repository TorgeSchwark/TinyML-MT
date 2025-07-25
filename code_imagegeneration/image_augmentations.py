from PIL import ImageEnhance, ImageDraw, ImageFilter
import numpy as np
import random
from PIL import Image


def apply_brightness(image, factor_range=(0.7, 1.3)):
    """
    Apply random brightness enhancement to an image.
    """
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)

def apply_contrast(image, factor_range=(0.7, 1.3)):
    """
    Apply random contrast enhancement to an image.
    """
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)

def apply_color_shift(image, shift_range):
    """
    Verschiebt RGB-Kanäle getrennt nach shift_range:
    shift_range = {"r": (-30, 30), "g": (-10, 10), "b": (-5, 20)}
    """

    print(shift_range)
    
    img_np = np.array(image).astype(np.int16)  # vermeidet Überlauf
    r_shift = np.random.randint(*shift_range["r"])
    g_shift = np.random.randint(*shift_range["g"])
    b_shift = np.random.randint(*shift_range["b"])

    img_np[..., 0] = np.clip(img_np[..., 0] + r_shift, 0, 255)
    img_np[..., 1] = np.clip(img_np[..., 1] + g_shift, 0, 255)
    img_np[..., 2] = np.clip(img_np[..., 2] + b_shift, 0, 255)

    return Image.fromarray(img_np.astype(np.uint8))


def apply_blur(image, radius_range=(1, 3)):
    """Wendet Gaußsches Weichzeichnen mit zufälligem Radius an."""
    radius = random.uniform(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_rotation(image, angle_range=(-180, 180)):
    """
    Apply random rotation to an image.
    """
    angle = random.uniform(*angle_range)
    return image.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

def apply_noise(image, noise_std=10):
    """
    Apply random noise to non-transparent areas of an RGBA image, or to the whole image if RGB.
    """
    image_np = np.array(image)

    if image.mode == "RGBA":
        rgb = image_np[..., :3]
        alpha = image_np[..., 3]

        noise = np.random.normal(0, noise_std, rgb.shape).astype(np.int16)
        
        # Maske: wo alpha > 0 → kein voller Durchblick
        mask = alpha > 0
        mask = np.expand_dims(mask, axis=2)  # Shape (H, W, 1) für Broadcasting

        # Rauschen nur anwenden, wo alpha > 0
        rgb_noisy = np.clip(rgb + noise * mask, 0, 255).astype(np.uint8)

        combined = np.concatenate((rgb_noisy, alpha[..., np.newaxis]), axis=2)
        return Image.fromarray(combined, mode="RGBA")

    else:
        # Kein Alpha → auf gesamtes Bild anwenden
        noise = np.random.normal(0, noise_std, image_np.shape).astype(np.int16)
        noisy = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode=image.mode)


def apply_shadow(image, shadow_alpha_range=(30, 80), blur_radius=5):
    """
    Add a soft shadow (ellipse) to an image.
    """
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    x0 = random.randint(-10, image.size[0] // 2)
    y0 = random.randint(-10, image.size[1] // 2)
    x1 = random.randint(image.size[0] // 2, image.size[0] + 10)
    y1 = random.randint(image.size[1] // 2, image.size[1] + 10)
    shadow_color = (0, 0, 0, random.randint(*shadow_alpha_range))
    draw.ellipse([x0, y0, x1, y1], fill=shadow_color)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.alpha_composite(image.convert("RGBA"), shadow)

def apply_zoom(image, zoom_factor_range=(1.1, 1.5), crop_margin=0.2):
    """
    Apply random zoom to an image by cropping and then resizing.
    The zoom is done by cropping a portion of the image and resizing it to the original size.
    """
    width, height = image.size
    zoom_factor = random.uniform(*zoom_factor_range)
    
    # Calculate the cropping box
    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)
    
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # Resize the cropped image back to original size
    zoomed_image = cropped_image.resize((width, height), Image.LANCZOS)
    
    return zoomed_image
