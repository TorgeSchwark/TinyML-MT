import os

path_to_kaggle_dataset = "C:\Users\Torge\.cache\kagglehub\datasets\moltean\fruits\versions\11\fruits-360_dataset_original-size\fruits-360-original-size"
path_to = "./Dataset/"

Categorie_dict = {"apple": [1, ]}
leafout = {"hit": 1, "delicios": 2, "rotten": 1, "eggplant": 0, "3": 1}

def transform_dataset_kaggle():
    for entry in os.listdir(path_to_kaggle_dataset):
        if os.path.isdir(os.path.join(path_to_kaggle_dataset, entry)):
            subfolder_to = ""
            if entry == "Test" or entry == "Validation" or entry == "Training":
                subfolder_to = entry
            if subfolder_to != "":
                pass


