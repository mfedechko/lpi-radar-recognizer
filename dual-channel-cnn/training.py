import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.feature import hog
import numpy as np
import os
import cv2
from scipy.signal import stft
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Назви класів ===
modulation_labels = {
    1: "LFM", 2: "2FSK", 3: "4FSK", 4: "8FSK", 5: "Costas",
    6: "2PSK", 7: "4PSK", 8: "8PSK", 9: "Barker", 10: "Huffman",
    11: "Frank", 12: "P1", 13: "P2", 14: "P3", 15: "P4", 16: "Px",
    17: "ZadoffChu", 18: "T1", 19: "T2", 20: "T3", 21: "T4",
    22: "NM", 23: "Noise"
}

# === Генерація STFT спектрограм ===
def generate_stft_image(iq_signal, n_fft=256, nperseg=128):
    complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
    f, t, Zxx = stft(complex_signal, nperseg=nperseg, nfft=n_fft)
    stft_magnitude = np.abs(Zxx)
    return stft_magnitude

# === Збереження спектрограм у файли ===
def save_spectrogram_images(X_subset, LBL_subset, out_dir="spectrograms"):
    os.makedirs(out_dir, exist_ok=True)

    for idx, (signal, meta) in enumerate(tqdm(zip(X_subset, LBL_subset), total=len(X_subset))):
        cls_id = int(meta[0])
        snr = int(meta[1])
        mod_name = modulation_labels.get(cls_id, f"Class{cls_id}")

        # Побудова спектрограми
        stft_img = generate_stft_image(signal)
        stft_img = np.interp(stft_img, (stft_img.min(), stft_img.max()), (0, 255)).astype(np.uint8)
        stft_img = cv2.resize(stft_img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Формування імені файлу
        filename = f"{mod_name}_SNR{snr}_{idx:03d}.png"
        filepath = os.path.join(out_dir, filename)

        # Збереження зображення
        cv2.imwrite(filepath, stft_img)

    print(f"✅ Усі зображення збережено у '{out_dir}'")

# === ВИКЛИК ===
# save_spectrogram_images(X_subset, LBL_subset)  # ← Запусти після визначення змінних

# Додатковий код для тренування моделі залишено без змін нижче...

# Гіперпараметри
NUM_CLASSES = 12
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Вибрані класи
SELECTED_CLASSES = [16, 10, 1, 9, 18, 13, 2, 14, 6, 3, 12, 21]
CLASS_MAPPING = {cls_id: i for i, cls_id in enumerate(SELECTED_CLASSES)}

# === PyTorch Dataset ===
class LPIDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = CLASS_MAPPING[int(self.labels[idx, 0])]

        # 2D CNN вхід
        img_tensor = self.transform(img)

        # HOG + 1D CNN вхід
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=False)
        hog_tensor = torch.tensor(hog_features, dtype=torch.float)

        return img_tensor, hog_tensor, label

# === Dual-Channel CNN ===
class DualChannelCNN(nn.Module):
    def __init__(self, hog_input_dim):
        super().__init__()

        # HOG 1D CNN
        self.hog_branch = nn.Sequential(
            nn.Linear(hog_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # 2D CNN branch
        self.img_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.flatten = nn.Flatten()
        self.fc_merge = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x_img, x_hog):
        x1 = self.img_branch(x_img)
        x1 = self.flatten(x1)

        x2 = self.hog_branch(x_hog)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc_merge(x)
        return x

# === Тренування ===
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total, correct = 0, 0
    for x_img, x_hog, y in dataloader:
        x_img, x_hog, y = x_img.to(device), x_hog.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x_img, x_hog)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"Train Accuracy: {acc:.4f}")

# === Запуск ===
def run_training(images, lbls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LPIDataset(images, lbls)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sample_hog = hog(images[0], orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False)
    model = DualChannelCNN(hog_input_dim=len(sample_hog)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_model(model, dataloader, criterion, optimizer, device)

    return model