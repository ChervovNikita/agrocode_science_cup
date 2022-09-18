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
from torchvision.transforms import InterpolationMode
from torch import nn
from PIL import Image
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

device = "cuda" if torch.cuda.is_available() else "cpu"


class Identical(nn.Module):
    def forward(self, x):
        return x


def model_head_fc(path):
    model_ = torch.load(path, map_location=torch.device(device))
    model_.head.fc = Identical()
    return model_


def model_head(path):
    model_ = torch.load(path, map_location=torch.device(device))
    model_.head = Identical()
    return model_


def model_fc(path):
    model_ = torch.load(path, map_location=torch.device(device))
    model_.fc = Identical()
    return model_


def model_classifier(path):
    model_ = torch.load(path, map_location=torch.device(device))
    model_.classifier = Identical()
    return model_


models_data = []

# 1. beit_large_in22k
model = model_head('weights/beit_large_in22k.pth')
transform = transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
                                transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 2. beit_base_patch16_224_in22k
model = model_head('weights/beit_base_patch16_224_in22k.pth')
transform = transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
                                transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 3. beit_base_patch16_384
model = model_head('weights/beit_base_patch16_384.pth')
transform = transforms.Compose([transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
                                transforms.CenterCrop(384), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 4. beit_large_patch16_512
model = model_head('weights/beit_large_patch16_512.pth')
transform = transforms.Compose([transforms.Resize(512, interpolation=InterpolationMode.BILINEAR),
                                transforms.CenterCrop(512), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 5. tf_efficientnet_l2_ns
model = model_classifier('weights/tf_efficientnet_l2_ns.pth')
transform = transforms.Compose([transforms.Resize(512, interpolation=InterpolationMode.BILINEAR),
                                transforms.CenterCrop(512), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

# 6. deit3_large_patch16_384_in21ft1k
model = model_head('weights/deit3_large_patch16_384_in21ft1k.pth')
transform = transforms.Compose([transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
                                transforms.CenterCrop(384), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
models_data.append((model, transform))

for model, transform in models_data:
    model.eval()
    model = model.to(device)
