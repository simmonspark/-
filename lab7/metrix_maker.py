import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image as image
from utils import get_img_path
import functools
import os
from PIL import ImageFile
import json
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

    def __getitem__(self, idx):
        img = image.open(self.img_path[idx])
        img = img.convert('RGB')
        img = np.array(img)
        img = img / 255.0
        return torch.Tensor(img).float().to('cuda').permute(2,0,1)

    def __len__(self):
        return len(self.img_path)


save_dir = "/media/unsi/media/generative_tmp"
output_json_path = "./file_paths.json"
file_paths = []


def save_all_files(data_loader, prefix):
    for i, x in enumerate(tqdm(data_loader, desc=f"Saving {prefix} files")):
        x_path = os.path.join(save_dir, f"{prefix}_x{i + 1}.npy")
        np.save(x_path, x.cpu().numpy())
        file_paths.append({"x": x_path})


if __name__ == "__main__":
    train_path, test_path = get_img_path()

    # Dataset 및 DataLoader 초기화
    ds = Dataset(train_path)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    save_all_files(dl, "data")

    with open(output_json_path, "w") as f:
        json.dump(file_paths, f, indent=4)
