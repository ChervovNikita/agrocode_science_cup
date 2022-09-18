import numpy as np
from tqdm.auto import tqdm
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features(path, models_data):
    # return np.random.random(100)
    with torch.no_grad():
        features = []
        for model, tfm in models_data:
            image = Image.open(path).convert('RGB')
            image = tfm(image)

            image = image[None, :, :, :].to(device)

            pred = np.array(model(image).detach().cpu()).squeeze()
            features.append(pred)

        features = np.concatenate(features, axis=0)
        features /= np.linalg.norm(features)
    return features


def get_emb(data, models_data, base_dir):
    emb = []
    for idx in tqdm(data.idx):
        emb.append(extract_features(f'{base_dir}{idx}.png', models_data))
    emb = np.array(emb)
    return emb
