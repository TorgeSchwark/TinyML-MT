# Basket AEye

Welcome to our code for the shopping basket price detection project in TinyML! 

## Installation

All python modules and their respective version are in the environment.yml:

1. Create conda environment: `conda env create -f environment.yml`
2. Activate your environment and start exploring the code


## Repository structure

- The Dataset files can be found [here](https://huggingface.co/datasets/TorgeSchwark/TinyML-MT-data) on Huggingface where the respective zip files have to be unzipped. The zips have to be pulled via `git lfs pull`. This includes our custom dataset as well as the MVTec dataset.

- Code for recording and manipulating data can be found in [dataset-code](dataset-code)
- Code for training can be found in [training-code](training-code)
    - The training pipeline for our first attempt with one regression value can be found in [training.ipynb](training-code/training.ipynb)
    - The pipeline for the second approach with classes is in [train_classes.ipynb](training-code/train_classes.ipynb)
    
- Important checkpoints, inference code and more are also available

## Runs

Most of our runs were monitored with WandB and a report on our architecture search can be seen [here](https://wandb.ai/maats/TinyML-CartDetection/reports/CNN-Architecture-Search--VmlldzoxMDkwNzgzNg?accessToken=r33qvir9puixmshqukaatmrbo87icisgllb2cf5qdu680wohlizs6s2aa6jgwaeh)!
