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
def check_collision(objects, new_object, x, y, max_overlap=0.3):
    new_bbox = (x, y, x + new_object.size[0], y + new_object.size[1])

    for obj, (ox, oy) in objects:
        obj_bbox = (ox, oy, ox + obj.size[0], oy + obj.size[1])

        # Intersection bbox
        ix1 = max(new_bbox[0], obj_bbox[0])
        iy1 = max(new_bbox[1], obj_bbox[1])
        ix2 = min(new_bbox[2], obj_bbox[2])
        iy2 = min(new_bbox[3], obj_bbox[3])

        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            obj_area = obj.size[0] * obj.size[1]
            if intersection_area / obj_area > max_overlap:
                return True

    return False


def place_objects_on_background(background_image, object_images, object_counts, category_to_id, image_resolution=(500, 500)):
    # Skalieren des Hintergrundbildes auf gewünschte Auflösung
    background_image = background_image.resize(image_resolution, Image.LANCZOS)
    bg_width, bg_height = background_image.size

    n_objects = len(object_images)
    grid_size = math.ceil(math.sqrt(n_objects))  # z. B. bei 8 → 3x3 Gitter

    base_max_w = bg_width // grid_size
    base_max_h = bg_height // grid_size

    placed_objects = []
    yolo_labels = []

    for object_image, category in object_images:
        # Lies den Skalierungsfaktor aus dem Image-Objekt
        scale = object_image.info.get("custom_scale", 1.0)

        max_w = int(base_max_w * scale)
        max_h = int(base_max_h * scale)

        # Objekt skalieren
        scaled = scale_object(object_image, max_w, max_h)
        print("1", scaled.size[0], scaled.size[1])

        # Augmentierung
        scaled = augment_object(scaled)
        print("2", scaled.size[0], scaled.size[1])

        placed = False
        attempts = 0

        while not placed and attempts < 8:
            max_x = bg_width - scaled.size[0]
            max_y = bg_height - scaled.size[1]

            if max_x <= 0 or max_y <= 0:
                print(f"Object {category} too large for background, retrying with smaller scale.")
                scaled = scale_object(object_image, max_w, max_h)
                attempts += 1
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            if not check_collision(placed_objects, scaled, x, y):
                placed_objects.append((scaled, (x, y)))
                placed = True
                object_counts[category] += 1

                # YOLO Label
                class_id = category_to_id[category]
                center_x = (x + scaled.size[0] / 2) / bg_width
                center_y = (y + scaled.size[1] / 2) / bg_height
                width = scaled.size[0] / bg_width
                height = scaled.size[1] / bg_height

                yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            else:
                attempts += 1

        if not placed:
            print(f"Object {category} could not be placed after 8 attempts.")

    # Falls kein Objekt erfolgreich platziert wurde → rekursiv neu versuchen
    if not placed_objects:
        return False, False

    for obj, (x, y) in placed_objects:
        background_image.paste(obj, (x, y), obj)

    # apply_snn
    return background_image, yolo_labels

