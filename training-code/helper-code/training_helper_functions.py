import random
from PIL import Image
import numpy as np 
import os 
import ast
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime



def custom_augmentation(image, noise_range = [0,0]):
    """ 
    Currently adds noise randomly using a normal distribution. The a std deviation is piked between noise_range[0] and noise_range[1]  
    """
    noise_factor = random.uniform(noise_range[0], noise_range[1])
    image = add_random_noise(image, noise_factor)
    
    return image

def add_random_noise(image, noise_factor=0.1):
    """
    Adds random noise to the image
    """
    np_image = np.array(image)  
    noise = np.random.randn(*np_image.shape) * noise_factor  
    noisy_image = np_image + noise  
    noisy_image = np.clip(noisy_image, 0, 255)  
    return Image.fromarray(np.uint8(noisy_image))

def load_image_and_label_custom_dataset_regression(file_name, dataset_path, image_size):
    """ 
    Used for regression task on Custom Dataset. Returns for a given file_name the resized image and the price (Lable)
    """
    image_path = os.path.join(dataset_path, file_name)
    image = Image.open(image_path).resize(image_size)
    # Load Label
    txt_file = file_name.replace(".jpg", ".txt")
    txt_path = os.path.join(dataset_path, txt_file)
    label = None
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                try:
                    label = float(lines[1].split()[-1])
                except ValueError:
                    print(f"Error reading in the prices of the file: {txt_path}")
    if label is None:
        raise ValueError(f"unable to load label: {txt_file}")
    return image, label

def load_image_and_labels_custom_classify(file_name, dataset_path, image_size,  num_classes):
    """
    Used for classification task on Custom Dataset. Returns for a given file the resized image + the class count vector.
    """
    # Load Image
    image_path = os.path.join(dataset_path, file_name)
    image = Image.open(image_path).resize(image_size)
    image = np.array(image, dtype=np.float32)
    
    # Load Label
    txt_file = file_name.replace(".jpg", ".txt")
    txt_path = os.path.join(dataset_path, txt_file)
    label = np.zeros(num_classes, dtype=np.float32)  # Initialize label vector with zeros
    
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Objects:"):
                    objects_info = line.split("Objects:")[1].strip()  # eg. "{57: 1, 49: 1}"
                    objects_info = objects_info.strip("{}").split(",")
                    
                    for obj in objects_info:
                        class_id, count = map(int, obj.split(":"))
                        class_id -= 1
                        label[class_id] = count  # add the num of objects of each class in the vector 
                if line.startswith("Total Price:"):
                    # Wenn der Preis benötigt wird, kann er hier extrahiert werden
                    total_price = float(line.split("Total Price:")[1].strip())

    else:
        raise ValueError(f"Label-Datei nicht gefunden: {txt_file}")
    
    return image, label


def get_total_num_classes_custom(prices_file_path):
    """
    Opens the file with the class lables and searches for the highest class id (number of classes)
    """
    prices = {}
    with open(prices_file_path, "r") as f:
        for line in f:
            if ": " in line:
                _, dict_str = line.strip().split(": ", 1)
                try:
                    current_dict = ast.literal_eval(dict_str)
                    for key, value in current_dict.items():
                        if key in prices:
                            if prices[key] != value:
                                print(f"ERROR: Konflikt with key  {key}: "
                                    f"{prices[key]} vs {value}")
                        else:
                            prices[key] = value
                except Exception as e:
                    print(f"Error procesing the line {line}\n{e}")

    # fid the key with the highest ID
    max_id = max(prices.keys())
    return max_id


def get_combined_dict(prices_file_path):
    """ Concatenates the price dictionarys of all datasets found in prices_file_path"""
    prices = {}
    with open(prices_file_path, "r") as f:
        for line in f:
            # Extract the dictonary out of that line
            if ": " in line:
                _, dict_str = line.strip().split(": ", 1)
                try:
                    current_dict = ast.literal_eval(dict_str)
                    for key, value in current_dict.items():
                        if key in prices:
                            if prices[key] != value:
                                print(f"ERROR: Konflikt with key  {key}: "
                                    f"{prices[key]} vs {value}")
                        else:
                            prices[key] = value
                except Exception as e:
                  print(f"Error procesing the line {line}\n{e}")
    # fid the key with the highest ID
    return prices


def show_test_predictions(model, test_dataset, num_samples=20):
    model.eval()  # Set the model to evaluation mode
    samples = random.sample(range(len(test_dataset)), num_samples)  # Randomly select samples
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # Create a 4x5 grid for the images
    
    for i, idx in enumerate(samples):
        image, label = test_dataset[idx]  # Load the image and label
        image_tensor = image.unsqueeze(0).to(model.device)  # Add batch dimension and move to GPU/CPU
        
        # Generate the prediction
        with torch.no_grad():
            prediction = model(image_tensor).cpu().item()
        
        # Display the image
        ax = axes[i // 5, i % 5]
        ax.imshow(image.permute(1, 2, 0))  # Convert tensor image to HWC format
        ax.set_title(f"Label: {label:.2f}\nPrediction: {prediction:.2f}")
        ax.axis("off")
    
    plt.tight_layout()

    try:
        wandb.log({"predictions": wandb.Image(plt)})  # Log predictions to wandb
    except Exception as e:
        print(f"Couldn't log predictions to wandb: {e}")

    plt.show()

def configure_wandb(dataset_paths, batch_size, image_size, seed, additional_name="", tags=[], ):
    # Sort the dataset names alphabetically
    dataset_names = "-".join(sorted([os.path.basename(path) for path in dataset_paths]))
    group = "Regression-" + dataset_names
    name = dataset_names + additional_name + "_" + datetime.now().strftime("%d%b-%H:%M:%S")

    wandb_logger = WandbLogger(project="TinyML-CartDetection", group=group, name=name, tags=tags)
    wandb_logger.experiment.config.update({"batch_size": batch_size, "image_size": image_size, "seed": seed})  # Log additional hyperparameters

    return wandb_logger
