# ======================= SETUP (Colab + Imports) =======================
!pip install --quiet torch torchvision scikit-image matplotlib scikit-learn psutil

import os
import zipfile
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import cv2
import time
import psutil

# ======================= DOWNLOAD DATA FROM GCS =======================
GCS_ZIP_PATH = "gs://mfedechko-datasets/spectrograms.zip"
LOCAL_ZIP_PATH = "spectrograms.zip"
EXTRACT_DIR = "data/spectrograms"

!gsutil cp $GCS_ZIP_PATH $LOCAL_ZIP_PATH

with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("✅ Data extracted to:", EXTRACT_DIR)

# ======================= DATASET =======================
class LPISpectrogramDataset(Dataset):
    def __init__(self, image_paths, class_names, transform=None):
        self.image_paths = image_paths
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255.0, dtype=torch.float).unsqueeze(0)

        # HOG features
        hog_feat = hog(image.squeeze().numpy(), orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
        hog_feat = torch.tensor(hog_feat, dtype=torch.float)

        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_names.index(class_name)
        return image, hog_feat, label

# ======================= PREPARE DATA =======================
all_paths = glob.glob(f"{EXTRACT_DIR}/*/*.png")
class_names = sorted(os.listdir(EXTRACT_DIR))

train_paths, val_paths = train_test_split(all_paths, test_size=0.2, stratify=[os.path.basename(os.path.dirname(p)) for p in all_paths])

train_ds = LPISpectrogramDataset(train_paths, class_names)
val_ds = LPISpectrogramDataset(val_paths, class_names)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ======================= MODEL =======================
import torch.nn as nn

class DualChannelCNN(nn.Module):
    def __init__(self, hog_dim, num_classes):
        super().__init__()
        self.hog_branch = nn.Sequential(
            nn.Linear(hog_dim, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU()
        )

        self.img_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 256, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_hog):
        x1 = self.img_branch(x_img)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.hog_branch(x_hog)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

# ======================= TRAINING =======================
from sklearn.metrics import accuracy_score

def print_system_stats():
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"🧠 RAM Used: {ram.used / 1e9:.2f} GB / {ram.total / 1e9:.2f} GB")
    print(f"🖥️ CPU Usage: {cpu:.2f}%")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem_alloc = torch.cuda.memory_allocated(0) / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"⚡ GPU: {gpu} | Memory Used: {mem_alloc:.2f} GB / {mem_total:.2f} GB")


def train(model, train_loader, val_loader, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_acc, val_acc = [], []

    for epoch in range(epochs):
        print("\n===========================")
        print(f"🚀 Epoch {epoch+1}/{epochs} started")
        start_time = time.time()
        print_system_stats()

        model.train()
        y_true, y_pred = [], []
        for img, hog_feat, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            img, hog_feat, labels = img.to(device), hog_feat.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img, hog_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            y_true += labels.tolist()
            y_pred += outputs.argmax(1).tolist()
        acc = accuracy_score(y_true, y_pred)
        train_acc.append(acc)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for img, hog_feat, labels in val_loader:
                img, hog_feat, labels = img.to(device), hog_feat.to(device), labels.to(device)
                outputs = model(img, hog_feat)
                y_true += labels.tolist()
                y_pred += outputs.argmax(1).tolist()
        val_accuracy = accuracy_score(y_true, y_pred)
        val_acc.append(val_accuracy)

        duration = time.time() - start_time
        print(f"✅ Epoch {epoch+1} finished in {duration:.2f}s | Train Acc: {train_acc[-1]:.4f} | Val Acc: {val_accuracy:.4f}")

    return model, train_acc, val_acc, y_true, y_pred

# ======================= RUN TRAINING =======================
sample_hog = hog(cv2.imread(train_paths[0], cv2.IMREAD_GRAYSCALE), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False)
model = DualChannelCNN(hog_dim=len(sample_hog), num_classes=len(class_names))
model, train_acc, val_acc, y_true, y_pred = train(model, train_loader, val_loader)

# ======================= PLOTS =======================
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# ======================= CONFUSION MATRIX =======================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()