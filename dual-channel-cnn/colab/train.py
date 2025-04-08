# ✅ Скрипт з повним завантаженням всіх HOG-фіч у памʼять (для 15 класів)
# Перед запуском: переконайся, що є GPU і вистачає RAM (~20–30 ГБ)
# Додатково: автоматичне завантаження з Google Cloud Storage (GCS)
# Та збереження найкращої моделі назад у GCS

import os, pickle, time, zipfile, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import GPUtil

# GCS support
from google.colab import auth
from google.cloud import storage

# 🔧 Параметри
BATCH_SIZE = 256
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.15
HOG_DIR = "hog_features"
IMG_ZIP = "spectrograms_bin.zip"
IMG_DIR = "spectrograms_bin"
GCS_BUCKET = "mfedechko-datasets"
MODEL_NAME = "best_model.pth"

CLASS_NAMES = [
    "LFM", "4FSK", "Costas", "2PSK", "Barker", "Huffman",
    "Frank", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"
]

# 🔐 Авторизація та GCS клієнт
auth.authenticate_user()
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)

# 📦 Завантаження зображень
if not os.path.exists(IMG_ZIP):
    print("⬇️ Завантаження spectrograms_bin.zip з GCS...")
    bucket.blob(IMG_ZIP).download_to_filename(IMG_ZIP)

if not os.path.exists(IMG_DIR):
    print("📦 Розпаковка архіву...")
    with zipfile.ZipFile(IMG_ZIP, 'r') as zip_ref:
        zip_ref.extractall(IMG_DIR)

# 📦 Завантаження HOG-фіч
if not os.path.exists(HOG_DIR):
    os.makedirs(HOG_DIR)
    print("⬇️ Завантаження HOG-файлів з GCS...")
    for cls in CLASS_NAMES:
        blob_name = f"hog_features/{cls}.pkl"
        local_path = os.path.join(HOG_DIR, f"{cls}.pkl")
        if not os.path.exists(local_path):
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            print(f"✅ Завантажено {cls}.pkl")


# 📦 Dataset — з повним завантаженням всіх HOG-фіч у памʼять
class FullHOGDataset(Dataset):
    def __init__(self, image_root, hog_dir, transform=None):
        self.transform = transform
        self.encoder = LabelEncoder()
        self.encoder.fit(CLASS_NAMES)

        self.data = []

        for cls in CLASS_NAMES:
            hog_file = os.path.join(hog_dir, f"{cls}.pkl")
            if not os.path.exists(hog_file):
                print(f"[!] HOG-файл не знайдено: {hog_file}")
                continue
            with open(hog_file, "rb") as f:
                hog_data = pickle.load(f)

            for feat, fname, label in zip(hog_data["features"], hog_data["filenames"], hog_data["labels"]):
                match = re.search(r"_SNR(-?\d+)_", fname)
                if match and int(match.group(1)) >= -6:
                    image_path = os.path.join(image_root, cls, fname)
                    if os.path.exists(image_path):
                        self.data.append((image_path, torch.tensor(feat, dtype=torch.float32), label))

        print(f"📦 Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, hog_feat, label_str = self.data[idx]
        image = Image.open(image_path).convert("L")
        image = self.transform(image) if self.transform else torch.tensor(np.array(image) / 255.0).unsqueeze(0).float()
        label = self.encoder.transform([label_str])[0]
        return image, hog_feat, label


# 🧠 Модель
class DualChannelNet(nn.Module):
    def __init__(self, num_classes, hog_input_size):
        super().__init__()
        self.hog_fc = nn.Sequential(
            nn.Linear(hog_input_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.img_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.img_fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 14 * 14, 256))
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x_img, x_hog):
        hog = self.hog_fc(x_hog)
        img = self.img_fc(self.img_cnn(x_img))
        return self.classifier(torch.cat([hog, img], dim=1))


# 🚀 Тренування

def train():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = FullHOGDataset(IMG_DIR, HOG_DIR, transform)
    hog_input_size = dataset[0][1].shape[0]

    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Використовується пристрій:", device)

    model = DualChannelNet(num_classes=len(CLASS_NAMES), hog_input_size=hog_input_size).to(device)

    # Перевірка, де знаходиться модель
    for name, param in model.named_parameters():
        print(f"{name} → {param.device}")
        break  # достатньо одного шару

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    total_start = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔁 Epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()

        loop = tqdm(train_loader, desc="Train")
        for i, (images, hogs, labels) in enumerate(loop):
            images, hogs, labels = images.to(device), hogs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, hogs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=total_loss / (i + 1), acc=100 * correct / total)

        train_acc = 100 * correct / total
        print(f"✅ Train Loss: {total_loss:.2f}, Accuracy: {train_acc:.2f}%")

        # 🔍 Валідація
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, hogs, labels in val_loader:
                images, hogs, labels = images.to(device), hogs.to(device), labels.to(device)
                outputs = model(images, hogs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        print(f"📊 Val Loss: {val_loss:.2f}, Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_NAME)
            print(f"💾 Збережено найкращу модель (val acc: {val_acc:.2f}%)")
            # Завантаження назад у GCS
            model_blob = bucket.blob(MODEL_NAME)
            model_blob.upload_from_filename(MODEL_NAME)
            print(f"☁️ Модель збережено у GCS як {MODEL_NAME}")

        # 📊 Ресурси
        print(f"🧠 CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"🧠 GPU: {gpu.memoryUsed:.1f}MB / {gpu.memoryTotal:.1f}MB | Load: {gpu.load * 100:.1f}%")

        print(f"⏱️ Тривалість епохи: {(time.time() - start_time) / 60:.2f} хв")

    print(f"⏱️ Загальний час тренування: {(time.time() - total_start) / 60:.2f} хв")

    print("\n📉 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# ▶️ Запуск
train()
