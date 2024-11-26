import numpy as np
from torch.utils.data import Dataset
import torch
from utils import load_json
from PIL import ImageFile
import json
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = load_json(json_dir)

    def __getitem__(self, idx):
        x = np.load(self.json_dir[idx]["x"])

        return torch.Tensor(x).float().to('cuda')

    def __len__(self):
        return len(self.json_dir)
