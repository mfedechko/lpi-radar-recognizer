# This script is memory consuming since it loads all HOG data to RAM

# ✅ Повний Colab-сумісний скрипт
# Підтримує: фільтрацію по SNR >= -6, 15 вибраних класів, побудову графіків, confusion matrix, classification report

import os, zipfile, pickle, time, psutil, random, re
import numpy as np
import GPUtil
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from google.colab import auth
auth.authenticate_user()
from google.cloud import storage
client = storage.Client()

# ========== Налаштування ==========
BATCH_SIZE = 512
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
GCS_BUCKET = "mfedechko-datasets"
PKL_NAME = "hog_features_all.pkl"
ZIP_NAME = "spectrograms_bin.zip"

MIN_TRAINED_SNR = -6

# Обрані класи:
class_names = [
    "2PSK", "4FSK", "Barker", "Costas", "Frank",
    "Huffman", "LFM", "P1", "P2", "P3", "P4",
    "T1", "T2", "T3", "T4"
]

# ========== Завантаження ==========
if not os.path.exists(ZIP_NAME):
    print("Завантаження файлу spectrograms_bin.zipююю")
    client.bucket(GCS_BUCKET).blob(ZIP_NAME).download_to_filename(ZIP_NAME)
if not os.path.exists(PKL_NAME):
    print("Завантаження файлу hog_features_all.pkl")
    client.bucket(GCS_BUCKET).blob(PKL_NAME).download_to_filename(PKL_NAME)
print("Всі файли завантажено")
if not os.path.exists("spectrograms_bin"):
    print("Розпакування spectrograms_bin.zip")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall("spectrograms_bin")

with open(PKL_NAME, "rb") as f:
    hog_data_raw = pickle.load(f)

# 🔎 Фільтрація HOG-даних по class_names
filtered_features, filtered_labels, filtered_filenames = [], [], []
for feat, label, fname in zip(hog_data_raw["features"], hog_data_raw["labels"], hog_data_raw["filenames"]):
    if label in class_names:
        filtered_features.append(feat)
        filtered_labels.append(label)
        filtered_filenames.append(fname)

hog_data = {
    "features": filtered_features,
    "labels": filtered_labels,
    "filenames": filtered_filenames
}

hog_input_size = len(hog_data["features"][0])

# ========== Dataset ==========
class DualChannelDataset(Dataset):
    def __init__(self, image_root, hog_data, transform=None):
        self.transform = transform
        self.encoder = LabelEncoder()
        self.encoder.fit(hog_data["labels"])
        self.hog_map = {
            fname: (torch.tensor(feat, dtype=torch.float32), self.encoder.transform([label])[0])
            for feat, label, fname in zip(hog_data["features"], hog_data["labels"], hog_data["filenames"])
        }

        self.image_paths = []
        for class_dir in os.listdir(image_root):
            class_path = os.path.join(image_root, class_dir)
            if not os.path.isdir(class_path) or class_dir not in class_names:
                continue
            for fname in os.listdir(class_path):
                match = re.search(r"_SNR(-?\d+)_", fname)
                if not match or int(match.group(1)) < MIN_TRAINED_SNR:
                    continue
                if fname in self.hog_map:
                    self.image_paths.append((os.path.join(class_path, fname), fname))

        seen_classes = set(os.path.basename(os.path.dirname(p[0])) for p in self.image_paths)
        print(f"✅ Завантажені класи: {sorted(seen_classes)}")
        print(f"🖼️ Кількість зображень: {len(self.image_paths)}")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, fname = self.image_paths[idx]
        image = Image.open(image_path).convert("L")
        image = self.transform(image) if self.transform else torch.tensor(np.array(image) / 255.0).unsqueeze(0).float()
        hog_feat, label = self.hog_map[fname]
        return image, hog_feat, label

# ========== Модель ==========
class DualChannelNet(nn.Module):
    def __init__(self, num_classes=len(class_names), hog_input_size=hog_input_size):
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
def get_system_usage():
    ram = psutil.virtual_memory().used / 1024**3
    cpu = psutil.cpu_percent()
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        return ram, cpu, gpu.memoryUsed, gpu.load * 100
    return ram, cpu, 0, 0

def train_model_with_tracking():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = DualChannelDataset("spectrograms_bin", hog_data, transform)
    indices = list(range(len(dataset)))
    split = int(np.floor(VALIDATION_SPLIT * len(dataset)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)
    train_loader = DataLoader(Subset(dataset, indices[split:]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[:split]), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = "dual_channel_best.pt"
    patience, patience_counter = 5, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔁 Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        start_time = time.time()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc="Train", unit="batch")

        for images, hogs, labels in loop:
            images, hogs, labels = images.to(device), hogs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, hogs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=running_loss / (total / BATCH_SIZE), acc=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for images, hogs, labels in val_loader:
                images, hogs, labels = images.to(device), hogs.to(device), labels.to(device)
                outputs = model(images, hogs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"✅ Train: {train_acc:.2f}%, Loss: {train_loss:.4f} | Val: {val_acc:.2f}%, Loss: {val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            client.bucket(GCS_BUCKET).blob(best_path).upload_from_filename(best_path)
            print(f"💾 Збережено найкращу модель (Val acc: {best_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏸️ patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping.")
                break

        ram, cpu, gpu_mem, gpu_load = get_system_usage()
        print(f"🧠 CPU: {cpu:.1f}% | RAM: {ram:.2f} GB | GPU: {gpu_load:.1f}% | MEM: {gpu_mem:.1f} MB")
        print(f"⏱️ Тривалість епохи: {(time.time() - start_time)/60:.2f} хв")

    # === Графіки ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Confusion Matrix ===
    print("📊 Завантаження найкращої моделі для Confusion Matrix...")
    model.load_state_dict(torch.load(best_path))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, hogs, labels in val_loader:
            images, hogs = images.to(device), hogs.to(device)
            outputs = model(images, hogs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Validation  Set)")
    plt.show()

    print("\n📋 Classification Report:\n")
    print(classification_report(all_true, all_preds, target_names=class_names))

# ▶️ Запуск
train_model_with_tracking()
