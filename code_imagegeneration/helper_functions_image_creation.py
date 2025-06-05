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
from datetime import datetime

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


def scale_object(object_image, area_per_image, custom_scale, size_variance=(0.9, 1.0)):
    """
    Scales object_image to match a target area (in pixels), preserving aspect ratio.
    - area_per_image: desired area in pixels²
    - custom_scale: base scale multiplier (e.g. from class-wise scale.txt)
    - size_variance: random range to vary object size within a class 
    """

    obj_width, obj_height = object_image.size
    aspect_ratio = obj_width / obj_height
   
    # Ideal height based on desired area and aspect ratio
    base_height = math.sqrt(area_per_image / aspect_ratio)
    base_width = base_height * aspect_ratio

    # Apply random variation and custom scaling
    variation = random.uniform(size_variance[0], size_variance[1])
    final_width = int(base_width * custom_scale * variation)
    final_height = int(base_height * custom_scale * variation)

    return object_image.resize((final_width, final_height), Image.LANCZOS)


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

def add_to_collision_grid(grid_indices, occupancy_grid, objects_positions, w, h, x, y, allowed_overlap_per_side, cell_size=10):
    grid_w = int((w*(1-2*allowed_overlap_per_side)) // cell_size)
    grid_h = int((h*(1-2*allowed_overlap_per_side)) // cell_size)
    grid_x = int((x+(w*allowed_overlap_per_side)) // cell_size)
    grid_y = int((y+(h*allowed_overlap_per_side)) // cell_size)

    num_cells_x, num_cells_y = occupancy_grid.shape

    objects_positions.append(((grid_x, grid_y),(grid_w, grid_h)))

    to_remove = set()

    for ind_x in range(grid_x, grid_x + grid_w):
        for ind_y in range(grid_y, grid_y + grid_h):
            occupancy_grid[ind_x, ind_y] = True
            idx_1d = int(ind_y * num_cells_x + ind_x) 
            to_remove.add(idx_1d)

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

def add_object_padding(occupancy_grid_object, grid_indices, objects_positions, curr_object_grid_size):
    ## TODO !!!! needs rework! 
    num_cells_x, num_cells_y = occupancy_grid_object.shape
    print("shape:", num_cells_x, num_cells_y )
    to_remove = set()

    def check_pos_and_add(x_pos, y_pos):
        if 0 <= x_pos < num_cells_x and 0 <= y_pos < num_cells_y:
            occupancy_grid_object[x_pos, y_pos] = True
            idx_1d = int(y_pos * num_cells_x + x_pos) 
            to_remove.add(idx_1d)

    for (pos_x, pos_y), (width_obj, height_obj) in objects_positions:
        for bottom_x in range(0, width_obj):
            for bottom_y in range(0, curr_object_grid_size[1]):
                nx = pos_x + bottom_x
                ny = pos_y - bottom_y
                check_pos_and_add(nx, ny)
        for bottom_x in range(curr_object_grid_size[0]):
            for bottom_y in range(curr_object_grid_size[1]):
                nx = pos_x - bottom_x
                ny = pos_y - bottom_y
                check_pos_and_add(nx, ny)
        for y_side in range(0, height_obj):
            for x_side in range(0, curr_object_grid_size[0]):
                nx = pos_x - x_side
                ny = pos_y + y_side
                check_pos_and_add(nx, ny)

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

def equals(occupancy_grid, grid_index_object):
    num_cells_x, num_cells_y = occupancy_grid.shape
    expected_indices = set()

    for x in range(num_cells_x):
        for y in range(num_cells_y):
            if not occupancy_grid[x, y]:
                index = y * num_cells_x + x
                expected_indices.add(index)

    actual_indices = set(grid_index_object)
    difference = expected_indices.symmetric_difference(actual_indices)

    return len(difference)





def place_objects_on_background(background_image, object_images, object_counts, category_to_id, image_resolution=(500, 500), show = False):
    # Determines the size one object can take more or less fixed since we always have same camera distance
    max_object_amount = 17+ random.randint(-2,2)

    cell_size = 2
    max_shadow_offset = 10
    boarder_size_percent = 0.1
    allowed_overlap_per_side = 0.1 

    num_cells_x = image_resolution[0] // cell_size
    num_cells_y = image_resolution[1] // cell_size
    
    boarder_x = int(num_cells_x * boarder_size_percent)
    boarder_y = int(num_cells_y * boarder_size_percent)

    num_cells_total = num_cells_x * num_cells_y

    grid_indices = list(range(num_cells_total))
    objects_positions = []
    occupancy_grid = np.zeros((num_cells_x, num_cells_y), dtype=bool)

    delete_boarders(grid_indices, occupancy_grid, boarder_x, boarder_y)

    background_image = background_image.resize(image_resolution, Image.LANCZOS)
                
    shadow_offset = (random.randint(-max_shadow_offset, max_shadow_offset), random.randint(-max_shadow_offset, max_shadow_offset))

    area_per_image = ((image_resolution[0]*(1-2*boarder_size_percent)) * (image_resolution[1]*1-2*boarder_size_percent)) // max_object_amount

    placed_objects = False
    placed_object_with_shadow = []
    yolo_labels = []

    for object_image, category in object_images:

        custom_scale = object_image.info.get("custom_scale", 1.0)

        # Save original size for YOLO before applying shadow
        scaled_obj = scale_object(object_image, area_per_image, custom_scale)
        scaled_obj = augment_object(scaled_obj)
        scaled_w, scaled_h = scaled_obj.size


        if not grid_indices:
            print("Keine freien Zellen mehr verfügbar.")
            break  # oder return
            
        occupancy_grid_object = copy.copy(occupancy_grid)
        grid_index_object = copy.copy(grid_indices)

        curr_object_grid_size = (int(scaled_w//cell_size), int(scaled_h//cell_size))
        add_object_padding(occupancy_grid_object, grid_index_object, objects_positions, curr_object_grid_size)

        add_boarder_padding(occupancy_grid_object, grid_index_object, curr_object_grid_size, boarder_x, boarder_y)

        print("um: ", equals(occupancy_grid_object, grid_index_object) ,"verschieden")

        # Zufällige freie Zelle wählen
        if not grid_index_object:
                print("Keine freien Zellen mehr verfügbar nach padding.")
                break  # oder return

        grid_index = random.choice(grid_index_object)
        grid_x = grid_index % num_cells_x
        grid_y = grid_index // num_cells_x

        x = grid_x * cell_size
        y = grid_y * cell_size

        add_to_collision_grid(grid_indices, occupancy_grid, objects_positions, scaled_w, scaled_h, x, y, allowed_overlap_per_side, cell_size=cell_size)

        with_shadow = add_realistic_shadow_and_light(scaled_obj, shadow_offset=shadow_offset)

        placed_objects = True
        placed_object_with_shadow.append((with_shadow, (x - with_shadow.size[0]//4, y - with_shadow.size[1]//4)))


        center_w, center_h = scaled_w // 2, scaled_h // 2

        placed = True
        object_counts[category] += 1

        # YOLO Label (nur bezogen auf Objektgröße, nicht Schatten)
        class_id = category_to_id[category]
        center_x = (x + center_w) / image_resolution[0]
        center_y = (y + center_h) / image_resolution[1]
        width = scaled_w / image_resolution[0]
        height = scaled_h / image_resolution[1]

        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        

    if not placed_objects:
        return False, False

    for obj, (x, y) in placed_object_with_shadow:
        # Paste schneidet automatisch Schatten ab, wenn er übersteht
        background_image.paste(obj, (x, y), obj)

    return background_image, yolo_labels


