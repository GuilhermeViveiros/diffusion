import os
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_FOLDER = "images"
os.makedirs(IMG_FOLDER, exist_ok=True)
