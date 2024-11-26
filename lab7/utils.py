import cv2
import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm


def get_img_path():
    img_dir = '/media/unsi/media/data/archive/celeba_hq_256'
    img_full_path = []

    for file_path, _, file_name in tqdm(os.walk(img_dir)):
        for name in file_name:
            img_name = os.path.splitext(name)[0] + '.jpg'
            img_path = os.path.join(img_dir, img_name)
            img_full_path.append(img_path)

    return img_full_path[:int(len(img_full_path) * 0.95)], img_full_path[int(len(img_full_path) * 0.95):]

if __name__ == '__main__':
    train,test = get_img_path()