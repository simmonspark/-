from PIL import ImageFile
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch
from utils import load_json


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = load_json(json_dir)
        self.data = []
        for idx in tqdm(range(len(self.json_dir)), desc="Loading Data into Memory"):
            x = np.load(self.json_dir[idx]["x"])
            self.data.append(x)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]).float().to('cuda')

    def __len__(self):
        # 데이터셋 길이 반환
        return len(self.data)

