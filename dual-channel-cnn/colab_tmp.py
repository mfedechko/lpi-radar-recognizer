# ✅ ПОВНИЙ СКРИПТ ДЛЯ GOOGLE COLAB: Training LPI-Net з ZIP архіву, моніторингом і оцінкою

from google.colab import auth
auth.authenticate_user()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time
import psutil
import GPUtil
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- ШЛЯХИ І ПАРАМЕТРИ ---
GCS_ZIP_PATH = "gs://mfedechko-datasets/deep_radar_2022.zip"
LOCAL_ZIP_PATH = "/content/deep_radar_2022.zip"
EXTRACT_DIR = "/content/dataset"
CHECKPOINT_DIR = "/content/checkpoints"
LOG_DIR = "/content/logs"
GCS_SAVE_PATH = "gs://mfedechko-datasets/models/lpinet_colab/"
IMG_SIZE = (128, 128)
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_EPOCHS = 45
LEARNING_RATE_DECAY_FACTOR = 0.1
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- ЗАВАНТАЖЕННЯ ТА РОЗПАКУВАННЯ ZIP ---
!gsutil cp $GCS_ZIP_PATH $LOCAL_ZIP_PATH
!unzip -q $LOCAL_ZIP_PATH -d $EXTRACT_DIR

# --- ПРИСТРІЙ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Використовується пристрій:", device)

# --- ТРАНСФОРМАЦІЇ ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- ЗАВАНТАЖЕННЯ ДАНИХ ---
dataset_path = os.path.join(EXTRACT_DIR, "deep_radar_2022")
dataset = ImageFolder(root=dataset_path, transform=transform)
NUM_CLASSES = len(dataset.classes)
print(f"🔎 Виявлено {NUM_CLASSES} класів: {dataset.classes}")

val_size = int(len(dataset) * VAL_SPLIT)
test_size = int(len(dataset) * TEST_SPLIT)
train_size = len(dataset) - val_size - test_size
train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# --- МОДЕЛЬ ---
class LPINet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(LPINet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.conv2b = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2a = nn.BatchNorm2d(64)
        self.bn2b = nn.BatchNorm2d(64)
        self.elu2a = nn.ELU()
        self.elu2b = nn.ELU()

        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.elu3 = nn.ELU()

        self.conv4a = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.conv4b = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn4a = nn.BatchNorm2d(64)
        self.bn4b = nn.BatchNorm2d(64)
        self.elu4a = nn.ELU()
        self.elu4b = nn.ELU()

        self.conv5 = nn.Conv2d(192, 64, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.elu5 = nn.ELU()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.elu1(self.bn1(self.conv1(x))))
        x1 = self.elu2a(self.bn2a(self.conv2a(x)))
        x2 = self.elu2b(self.bn2b(self.conv2b(x)))
        x = torch.cat((x1, x2), dim=1)
        x = self.elu3(self.bn3(self.conv3(x)))
        x1 = self.elu4a(self.bn4a(self.conv4a(x)))
        x2 = self.elu4b(self.bn4b(self.conv4b(x)))
        x = torch.cat((x1, x2, x), dim=1)
        x = self.elu5(self.bn5(self.conv5(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- ТРЕНУВАННЯ ---
model = LPINet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LEARNING_RATE_DECAY_EPOCHS, gamma=LEARNING_RATE_DECAY_FACTOR)
best_val_acc = 0
epochs_no_improve = 0
early_stop_patience = 20
train_log = []

for epoch in range(EPOCHS):
    print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()
    loop = tqdm(train_loader, desc="Train", unit="batch")

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=running_loss / (total / BATCH_SIZE), acc=100 * correct / total)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    scheduler.step()

    print(f"✅ Train: {train_loss:.4f}, {train_acc:.2f}% | Val: {val_loss:.4f}, {val_acc:.2f}%")
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"🚀 GPU Usage: {gpu.load * 100:.1f}% | Memory: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} MB")
    print(f"🧠 CPU: {psutil.cpu_percent():.1f}% | RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")
    print(f"⏱️ Тривалість епохи: {(time.time() - start_time)/60:.2f} хв")

    train_log.append({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        !gsutil cp {os.path.join(CHECKPOINT_DIR, "best_model.pth")} {GCS_SAVE_PATH}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print("🛑 Early stopping activated")
            break

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth"))
    !gsutil cp {os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")} {GCS_SAVE_PATH}

# --- ТЕСТУВАННЯ ---
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print("\n📊 Final Test Evaluation:")
print(classification_report(test_labels, test_preds, target_names=dataset.classes))
conf_matrix = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
conf_matrix_path = os.path.join(LOG_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.show()
!gsutil cp {conf_matrix_path} {GCS_SAVE_PATH}

# --- ГРАФІК НАВЧАННЯ ---
losses = [x['train_loss'] for x in train_log]
val_losses = [x['val_loss'] for x in train_log]
accs = [x['train_acc'] for x in train_log]
val_accs = [x['val_acc'] for x in train_log]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

plt.suptitle("LPI-Net Training Progress")
plt.tight_layout()
plot_path = os.path.join(LOG_DIR, "training_plot.png")
plt.savefig(plot_path)
plt.show()
!gsutil cp {plot_path} {GCS_SAVE_PATH}