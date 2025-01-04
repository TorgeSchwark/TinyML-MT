import kagglehub
from roboflow import Roboflow


# Download latest version
path = kagglehub.dataset_download("moltean/fruits")

print("Path to dataset files:", path)

# Den API-Key aus der Datei lesen
with open("api_key.txt", "r") as file:
    api_key = file.read().strip()  

# Use roboflow with the API key
rf = Roboflow(api_key=api_key)
project = rf.workspace("vegetables").project("vegetables-el4g6")
version = project.version(1)
dataset = version.download("darknet")
                 