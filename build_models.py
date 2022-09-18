import cv2
import numpy as np
import math
import pandas as pd
from tqdm.auto import tqdm
import random
import string
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
import timm
import torch
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

device = "cuda" if torch.cuda.is_available() else "cpu"


class Identical(nn.Module):
    def forward(self, x):
        return x


def model_head_fc(path):
    model = torch.load(path, map_location=torch.device(device))
    model.head.fc = Identical()
    return model


def model_head(path):
    model = torch.load(path, map_location=torch.device(device))
    model.head = Identical()
    return model


def model_fc(path):
    model = torch.load(path, map_location=torch.device(device))
    model.fc = Identical()
    return model


models_data = []

# 1. beit_large_in22k
model = model_head('weights/beit_large_in22k.pth')
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 2. beit_base_patch16_224_in22k
model = model_head('weights/beit_base_patch16_224_in22k.pth')
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 3. beit_base_patch16_384
model = model_head('weights/beit_base_patch16_384.pth')
transform = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(384), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 4. beit_large_patch16_512
model = model_head('weights/beit_large_patch16_512.pth')
transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# models_data = [models_data[1]]
for model, transform in models_data:
    model.eval()
    model = model.to(device)
