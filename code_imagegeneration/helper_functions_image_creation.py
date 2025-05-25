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
import copy

from image_augmentations import *

def augment_object(image, brightness_prob=0.8, brightness_range=(0.7, 1.3), contrast_prob=0.8, contrast_range=(0.7, 1.3), rotation_prob=1, rotation_range=(-90, 90),
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
def check_collision(occupancy_grid, new_w, new_h, x, y, max_overlap=50, grid_size=10):
    new_bbox = (x, y, x + new_w, y + new_h)
    new_area = new_w * new_h
    position_on_grid = (x // grid_size, y // grid_size)
    grid_w = new_w // grid_size
    grid_h = new_h // grid_size
    grid_area = grid_w * grid_h
    count = 0


    for ind_x in range(position_on_grid[0], position_on_grid[0]+ grid_w):
        for ind_y in range(position_on_grid[1],position_on_grid[1]+grid_h):
            if ind_x >= 0 and ind_y >= 0 and ind_x < occupancy_grid.shape[0] and ind_y < occupancy_grid.shape[1]:
                if occupancy_grid[ind_x, ind_y]:
                    count += 1
                    if count > max_overlap:
                        return True
            else:
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

def add_to_collision_grid(grid_indices, occupancy_grid, object_position, w, h, x, y, grid_size=10):
    grid_x = x // grid_size
    grid_y = y // grid_size
    grid_w = w // grid_size
    grid_h = h // grid_size

    object_position.append((grid_x, grid_y))

    to_remove = set()

    for ind_x in range(grid_x, grid_x + grid_w):
        for ind_y in range(grid_y, grid_y + grid_h):
            occupancy_grid[ind_x, ind_y] = True
            to_remove.add((ind_x, ind_y))

    grid_indices[:] = [idx for idx in grid_indices if idx not in to_remove]

            
def delete_boarders(grid_indices, occupancy_grid, boarder_x, boarder_y):
    num_cells_x, num_cells_y = occupancy_grid.shape
    to_remove = set()

    # Ränder: links und rechts
    for ind_x in range(num_cells_x):
        for ind_y in range(num_cells_y):
            if ind_x < boarder_x or ind_x >= num_cells_x - boarder_x \
            or ind_y < boarder_y or ind_y >= num_cells_y - boarder_y:
                occupancy_grid[ind_x, ind_y] = True
                to_remove.add(ind_y * num_cells_x + ind_x)  # 1D Index

    # Entferne belegte Indizes aus der Liste
    grid_indices[:] = [idx for idx in grid_indices if idx not in to_remove]

def add_object_padding(occupancy_grid_object, grid_indices, object_positions, object_size):
    num_cells_x, num_cells_y = occupancy_grid_object.shape
    to_remove = set()

    for pos_x, pos_y in object_positions:
        for dx in range(object_size[0]):
            for dy in range(object_size[1]):
                nx = pos_x - dx
                ny = pos_y - dy
                if 0 <= nx < num_cells_x and 0 <= ny < num_cells_y:
                    occupancy_grid_object[nx, ny] = True
                    # 1D index berechnen und merken
                    idx_1d = int(ny * num_cells_x + nx) # Achte auf Reihenfolge (y * width + x)
                    to_remove.add(idx_1d)

    # grid_indices ist eine Liste von 1D-Indices: entferne belegte Zellen
    grid_indices[:] = [idx for idx in grid_indices if idx not in to_remove]

def add_boarder_padding(occupancy_grid_object, grid_index_object, object_grid_size, boarder_x, boarder_y):
    num_cells_x, num_cells_y = occupancy_grid_object.shape
    to_remove = set()

    # Bereich links und rechts
    for x in range(num_cells_x):
        for y in range(num_cells_y):
            # Prüfen, ob das Objekt mit seiner Größe + Padding den Rand überlappen würde
            if x + object_grid_size[0] > num_cells_x - boarder_x or \
               y + object_grid_size[1] > num_cells_y - boarder_y:
                occupancy_grid_object[x, y] = True
                idx_1d = y * num_cells_x + x
                to_remove.add(idx_1d)

    # Entferne alle betroffenen Indizes aus den verfügbaren Zellen
    grid_index_object[:] = [idx for idx in grid_index_object if idx not in to_remove]




def place_objects_on_background(background_image, object_images, object_counts, category_to_id, image_resolution=(500, 500), show = False):

    cell_size = 10
    num_cells_x = image_resolution[0] // 10
    num_cells_y = image_resolution[1] // 10
    n_objects = len(object_images)

    
    boarder_x = int(num_cells_x * 0.1)
    boarder_y = int(num_cells_y * 0.1)

    num_cells_total = num_cells_x * num_cells_y

    overlap_area = (num_cells_total*cell_size) // ((n_objects+3) * 50)

    grid_indices = list(range(num_cells_total))
    object_position = []
    occupancy_grid = np.zeros((num_cells_x, num_cells_y), dtype=bool)

    delete_boarders(grid_indices, occupancy_grid, boarder_x, boarder_y)

    background_image = background_image.resize(image_resolution, Image.LANCZOS)
                
    shadow_offset = (random.randint(-10, 10), random.randint(-10, 10))

    bg_width, bg_height = background_image.size

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

        overlap_area_on_size = max_w/cell_size* max_h/cell_size // 4 # max 20%overlap
        if overlap_area_on_size < overlap_area:
            overlap_area = overlap_area_on_size
            print(f"Overlap area {overlap_area} is too large for object {category}.")

        # Save original size for YOLO before applying shadow
        scaled_obj = scale_object(object_image, max_w, max_h)
        scaled_obj = augment_object(scaled_obj)
        original_w, original_h = scaled_obj.size

        placed = False
        attempts = 0

        if not grid_indices:
                print("Keine freien Zellen mehr verfügbar.")
                break  # oder return
            
        occupancy_grid_object = copy.copy(occupancy_grid)
        grid_index_object = copy.copy(grid_indices)

        padding_size_w = original_w *0.7
        padding_size_h = original_h *0.7
        object_grid_size = (int(padding_size_w//cell_size), int(padding_size_h//cell_size))
        add_object_padding(occupancy_grid_object, grid_index_object, object_position, object_grid_size)
        add_boarder_padding(occupancy_grid_object, grid_index_object, object_grid_size, boarder_x, boarder_y)
        # Zufällige freie Zelle wählen
        if not grid_index_object:
                print("Keine freien Zellen mehr verfügbar nach padding.")
                break  # oder return
        print(len(grid_index_object)," places to be ")

        while not placed and attempts < 20:

            grid_index = random.choice(grid_index_object)
            grid_x = grid_index % num_cells_x
            grid_y = grid_index // num_cells_x

            cell_size = 10
            x = grid_x * cell_size
            y = grid_y * cell_size


            center_w, center_h = original_w // 2, original_h // 2
            if not check_collision(occupancy_grid, original_w, original_h, x , y, max_overlap= overlap_area):

                # Jetzt erst Schatten generieren – damit der Schatten beliebig übersteht
                add_to_collision_grid(grid_indices, occupancy_grid, object_position, original_w, original_h, x, y)
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


