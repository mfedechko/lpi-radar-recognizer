# ✅ Colab-скрипт (HOG читаються з диску по одному)
# Працює з малим об'ємом RAM. Перед запуском — кешовані class-wise .pkl у 'hog_features/'

import os, zipfile, pickle, time, psutil, re
import random
import numpy as np
from PIL import Image
import GPUtil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

from google.colab import auth
auth.authenticate_user()
from google.cloud import storage
client = storage.Client()

# 🔧 Налаштування
BATCH_SIZE = 256
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
GCS_BUCKET = "mfedechko-datasets"
ZIP_NAME = "spectrograms_bin.zip"
HOG_DIR = "hog_features"
class_names = ["2PSK", "4FSK", "Barker", "Costas", "Frank", "Huffman", "LFM", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

MIN_SNR = -6

# 📦 Завантаження spectrograms_bin з GCS
if not os.path.exists(ZIP_NAME):
    print("Завантаження spectrograms_bin.zip")
    client.bucket(GCS_BUCKET).blob(ZIP_NAME).download_to_filename(ZIP_NAME)
if not os.path.exists("spectrograms_bin"):
    print("Розпакування архіву")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall("spectrograms_bin")

# 📦 Завантаження HOG-фічей з GCS
if not os.path.exists(HOG_DIR):
    print("Завантаження HOG файлів")
    os.makedirs(HOG_DIR)
    bucket = client.bucket(GCS_BUCKET)
    blobs = bucket.list_blobs(prefix=f"{HOG_DIR}/")
    for blob in blobs:
        if blob.name.endswith(".pkl"):
            local_path = os.path.join(HOG_DIR, os.path.basename(blob.name))
            blob.download_to_filename(local_path)

# ========== Dataset ==========
class DualChannelDataset(Dataset):
    def __init__(self, image_root, hog_dir, transform=None):
        self.transform = transform
        self.encoder = LabelEncoder()
        self.encoder.fit(class_names)
        self.hog_dir = hog_dir
        self.image_paths = []

        for cls in class_names:
            class_path = os.path.join(image_root, cls)
            hog_file = os.path.join(hog_dir, f"{cls}.pkl")
            if not os.path.exists(class_path) or not os.path.exists(hog_file):
                continue

            with open(hog_file, "rb") as f:
                data = pickle.load(f)
            class_filenames = set(data["filenames"])

            for fname in os.listdir(class_path):
                if fname not in class_filenames:
                    continue
                match = re.search(r"_SNR(-?\d+)_", fname)
                if match and int(match.group(1)) >= MIN_SNR:
                    self.image_paths.append((os.path.join(class_path, fname), fname, cls))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, fname, cls = self.image_paths[idx]
        image = Image.open(image_path).convert("L")
        image = self.transform(image) if self.transform else torch.tensor(np.array(image) / 255.0).unsqueeze(0).float()

        hog_file = os.path.join(self.hog_dir, f"{cls}.pkl")
        with open(hog_file, "rb") as f:
            data = pickle.load(f)
        i = data["filenames"].index(fname)
        hog_feat = torch.tensor(data["features"][i], dtype=torch.float32)
        label = self.encoder.transform([data["labels"][i]])[0]

        return image, hog_feat, label

# ========== Модель ==========
class DualChannelNet(nn.Module):
    def __init__(self, num_classes, hog_input_size):
        super().__init__()
        self.hog_cnn = nn.Sequential(
            nn.Conv1d(1, 8, 5), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(8, 16, 3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, hog_input_size)
            out_size = self.hog_cnn(dummy).view(1, -1).shape[1]
        self.hog_fc = nn.Linear(out_size, 256)

        self.img_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.img_fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 14 * 14, 256))
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x_img, x_hog):
        hog = self.hog_cnn(x_hog.unsqueeze(1)).view(x_hog.size(0), -1)
        hog = self.hog_fc(hog)
        img = self.img_fc(self.img_cnn(x_img))
        return self.classifier(torch.cat([hog, img], dim=1))

# ========== Тренування ==========
def train():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = DualChannelDataset("spectrograms_bin", HOG_DIR, transform)
    hog_input_size = dataset[0][1].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualChannelNet(num_classes=len(class_names), hog_input_size=hog_input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * VALIDATION_SPLIT)
    train_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔁 Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for images, hogs, labels in tqdm(train_loader, desc="Train"):
            images, hogs, labels = images.to(device), hogs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, hogs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"✅ Train Loss: {running_loss:.2f}, Accuracy: {acc:.2f}%")

        # 🔍 Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, hogs, labels in val_loader:
                images, hogs, labels = images.to(device), hogs.to(device), labels.to(device)
                outputs = model(images, hogs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"📊 Val Loss: {val_loss:.2f}, Accuracy: {val_acc:.2f}%")

        # 🧠 Ресурси
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"🧠 GPU: {gpus[0].memoryUsed:.1f}MB / {gpus[0].memoryTotal:.1f}MB | Load: {gpus[0].load*100:.1f}%")
        print(f"🧠 CPU: {psutil.cpu_percent():.1f}% | RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")
        print(f"⏱️ Тривалість епохи: {(time.time() - start_time)/60:.2f} хв")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, dataset, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, hogs, labels in dataloader:
            images, hogs = images.to(device), hogs.to(device)
            outputs = model(images, hogs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    label_names = dataset.encoder.classes_
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=2))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

train()
evaluate_model(model, val_loader, dataset, device)

