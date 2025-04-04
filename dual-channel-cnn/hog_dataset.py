from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import pickle
import numpy as np

class DualChannelDataset(Dataset):
    def __init__(self, image_root, hog_pkl_path, transform=None):
        self.image_root = image_root
        self.transform = transform

        # Завантажити HOG
        with open(hog_pkl_path, "rb") as f:
            hog_data = pickle.load(f)

        # Словник для швидкого доступу до HOG фіч
        self.hog_map = {
            fname: (torch.tensor(feat, dtype=torch.float32), label)
            for feat, label, fname in zip(hog_data["features"], hog_data["labels"], hog_data["filenames"])
        }

        # Створити список усіх доступних зображень, які є в hog_map
        self.image_paths = []
        for label_dir in os.listdir(image_root):
            class_path = os.path.join(image_root, label_dir)
            if not os.path.isdir(class_path):
                continue
            for file in os.listdir(class_path):
                if file in self.hog_map:
                    full_path = os.path.join(class_path, file)
                    self.image_paths.append((full_path, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, fname = self.image_paths[idx]
        image = Image.open(image_path).convert("L")  # бінарне зображення
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image) / 255.0).unsqueeze(0).float()  # [1, H, W]

        hog_feat, label = self.hog_map[fname]
        return image, hog_feat, label

dataset = DualChannelDataset(
    image_root="./spectrograms_bin",  # шлях до розпакованих зображень
    hog_pkl_path="hog_features_all.pkl"
)

img, hog, label = dataset[0]
print(img.shape)      # torch.Size([1, 224, 224])
print(hog.shape)      # torch.Size([26244]) (або інша, залежно від HOG)
print(label)          # Назва класу, напр. "LFM"