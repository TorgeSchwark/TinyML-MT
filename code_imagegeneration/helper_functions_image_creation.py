# Imports 
import os
from rembg import remove
from PIL import Image
import io
import random
import math

import numpy as np
from PIL import ImageEnhance, ImageFilter, ImageOps
from PIL import ImageDraw

from PIL import ImageEnhance, ImageDraw, ImageFilter
import numpy as np
import random

from image_augmentations import *

def augment_object(image, brightness_prob=0.8, brightness_range=(0.7, 1.3), contrast_prob=0.8, contrast_range=(0.7, 1.3), rotation_prob=0.5, rotation_range=(-15, 15),
                    noise_prob=0.3, noise_std=5, shadow_prob=0.5, shadow_alpha_range=(30, 80), shadow_blur_radius=5):
    
    if random.random() < brightness_prob: image = apply_brightness(image, brightness_range)

    if random.random() < contrast_prob: image = apply_contrast(image, contrast_range)

    if random.random() < rotation_prob: image = apply_rotation(image, rotation_range)

    if random.random() < noise_prob: image = apply_noise(image, noise_std)

    # if random.random() < shadow_prob: image = apply_shadow(image, shadow_alpha_range, shadow_blur_radius)

    return image


def augment_background(image, brightness_prob=0.9, brightness_range=(0.5, 1.5), 
                       contrast_prob=0.9, contrast_range=(0.5, 1.5), 
                       rotation_prob=0.7, rotation_range=(-30, 30), 
                       noise_prob=0.5, noise_std=5, 
                       shadow_prob=0.7, shadow_alpha_range=(50, 100), shadow_blur_radius=7, 
                       zoom_prob=0.6, zoom_range=(0.8, 1.2)):

    if random.random() < brightness_prob: 
        image = apply_brightness(image, brightness_range)

    if random.random() < contrast_prob: 
        image = apply_contrast(image, contrast_range)

    if random.random() < rotation_prob: 
        image = apply_rotation_fill(image, rotation_range)

    if random.random() < noise_prob: 
        image = apply_noise(image, noise_std)

    if random.random() < shadow_prob: 
        image = apply_shadow(image, shadow_alpha_range, shadow_blur_radius)

    return image

def apply_rotation_fill(image, rotation_range):
    image = image.convert("RGBA")
    angle = random.uniform(*rotation_range)
    orig_w, orig_h = image.size

    # 1. Rotieren mit expand=True (größerer Canvas, aber transparente Ecken)
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    # 2. Reingezoomt auf Originalformat (größer als nötig, um Ecken zu füllen)
    rotated_w, rotated_h = rotated.size

    # Zoomfaktor: So groß, dass Originalformat ohne Transparenz ausgeschnitten werden kann
    scale_w = orig_w*2 / rotated_w
    scale_h = orig_h*2 / rotated_h
    scale = max(scale_w, scale_h) * 1.05  # Etwas mehr als nötig, um Ränder sicher zu füllen

    new_w = int(rotated_w * scale)
    new_h = int(rotated_h * scale)
    resized = rotated.resize((new_w, new_h), resample=Image.BICUBIC)

    # 3. Von der Mitte ausschneiden
    left = (new_w - orig_w) // 2
    top = (new_h - orig_h) // 2
    cropped = resized.crop((left, top, left + orig_w, top + orig_h))

    return cropped



def scale_object(object_image, max_width, max_height):
    obj_width, obj_height = object_image.size
    aspect_ratio = obj_width / obj_height

    max_scale_w = max_width / obj_width
    max_scale_h = max_height / obj_height
    max_scale = min(max_scale_w, max_scale_h)

    scale_factor = random.uniform(0.9 * max_scale, max_scale)

    new_width = int(obj_width * scale_factor)
    new_height = int(obj_height * scale_factor)

    return object_image.resize((new_width, new_height), Image.LANCZOS)


# Check if new_object overlaps existing objects beyond allowed percentage
def check_collision(objects, new_w, new_h, x, y, max_overlap=0.4):
    new_bbox = (x, y, x + new_w, y + new_h)
    new_area = new_w * new_h

    for (ob_w, ob_h), (ox, oy) in objects:
        obj_bbox = (ox, oy, ox + ob_w, oy + ob_h)
        obj_area = ob_w * ob_h

        # Intersection bbox
        ix1 = max(new_bbox[0], obj_bbox[0])
        iy1 = max(new_bbox[1], obj_bbox[1])
        ix2 = min(new_bbox[2], obj_bbox[2])
        iy2 = min(new_bbox[3], obj_bbox[3])

        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            union_area = new_area + obj_area - intersection_area
            overlap_ratio = obj_area/ union_area

            if overlap_ratio > max_overlap:
                return True

    return False


def add_realistic_shadow_and_light(obj, shadow_offset=(4, 4), blur_radius=5,
                                   shadow_color=(0, 0, 0, 30)):
    w, h = obj.size
    canvas_w, canvas_h = w * 2, h * 2

    center_x = canvas_w // 2
    center_y = canvas_h // 2

    # Leerer Layer für Schatten und Objekt
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    object_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # 1. Alpha-Kanal holen
    obj_alpha = obj.split()[-1]

    # 2. Großes leeres Alphabild erstellen
    full_alpha = Image.new("L", (canvas_w, canvas_h), 0)
    alpha_pos = (center_x - w // 2 + shadow_offset[0], center_y - h // 2 + shadow_offset[1])
    full_alpha.paste(obj_alpha, alpha_pos)

    # 3. Schattenbild erzeugen
    shadow = Image.new("RGBA", (canvas_w, canvas_h), shadow_color)
    shadow.putalpha(full_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

    # 4. Objekt platzieren
    object_pos = (center_x - w // 2, center_y - h // 2)
    object_layer.paste(obj, object_pos, mask=obj)

    # 5. Kombinieren
    combined = Image.alpha_composite(shadow, object_layer)

    return combined



def place_objects_on_background(background_image, object_images, object_counts, category_to_id, image_resolution=(500, 500)):
    background_image = background_image.resize(image_resolution, Image.LANCZOS)
                
    shadow_offset = (random.randint(-10, 10), random.randint(-10, 10))

    bg_width, bg_height = background_image.size

    n_objects = len(object_images)
    grid_size = math.ceil(math.sqrt(n_objects))

    base_max_w = bg_width // grid_size
    base_max_h = bg_height // grid_size

    placed_objects = []
    placed_object_with_shadow = []
    yolo_labels = []

    for object_image, category in object_images:
        scale = object_image.info.get("custom_scale", 1.0)

        max_w = int(base_max_w * scale)
        max_h = int(base_max_h * scale)

        # Save original size for YOLO before applying shadow
        scaled_obj = scale_object(object_image, max_w, max_h)
        scaled_obj = augment_object(scaled_obj)
        original_w, original_h = scaled_obj.size

        placed = False
        attempts = 0

        while not placed and attempts < 8:
            max_x = bg_width - original_w
            max_y = bg_height - original_h

            if max_x <= 0 or max_y <= 0:
                print(f"Object {category} too large for background, retrying with smaller scale.")
                scaled_obj = scale_object(object_image, max_w, max_h)
                attempts += 1
                continue

            # Platzierungspunkt für das **Objekt**, nicht den Schatten
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            center_w, center_h = original_w // 2, original_h // 2
            if not check_collision(placed_objects, original_w, original_h, x , y):
                # Jetzt erst Schatten generieren – damit der Schatten beliebig übersteht
                with_shadow = add_realistic_shadow_and_light(scaled_obj, shadow_offset=shadow_offset)

                placed_objects.append(((original_w, original_h), (x, y)))
                placed_object_with_shadow.append((with_shadow, (x - with_shadow.size[0]//4, y - with_shadow.size[1]//4)))

                placed = True
                object_counts[category] += 1

                # YOLO Label (nur bezogen auf Objektgröße, nicht Schatten)
                class_id = category_to_id[category]
                center_x = (x + center_w) / bg_width
                center_y = (y + center_h) / bg_height
                width = original_w / bg_width
                height = original_h / bg_height

                yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            else:
                attempts += 1

        if not placed:
            print(f"Object {category} could not be placed after 8 attempts.")

    if not placed_objects:
        return False, False

    for obj, (x, y) in placed_object_with_shadow:
        # Paste schneidet automatisch Schatten ab, wenn er übersteht
        background_image.paste(obj, (x, y), obj)

    return background_image, yolo_labels


