import numpy as np
import os
import cv2
from scipy.signal import stft
from tqdm import tqdm
import scipy.io as sio
import h5py

# === Назви класів з опису Kaggle ===
modulation_labels = {
    1: "LFM", 2: "2FSK", 3: "4FSK", 4: "8FSK", 5: "Costas",
    6: "2PSK", 7: "4PSK", 8: "8PSK", 9: "Barker", 10: "Huffman",
    11: "Frank", 12: "P1", 13: "P2", 14: "P3", 15: "P4", 16: "Px",
    17: "ZadoffChu", 18: "T1", 19: "T2", 20: "T3", 21: "T4",
    22: "NM", 23: "Noise"
}

# === Функція побудови STFT спектрограми ===
def generate_stft_image(iq_signal, n_fft=256, nperseg=128):
    complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
    f, t, Zxx = stft(complex_signal, nperseg=nperseg, nfft=n_fft)
    stft_magnitude = np.abs(Zxx)
    return stft_magnitude

# === Завантаження .mat (HDF5) сигналів ===
def load_signals_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        signals = f['X_train'][:]  # (2, 1024, N)
        signals = signals.transpose(2, 1, 0)  # → (N, 1024, 2)
    return signals

# === Основна функція для обробки всього датасету ===
def save_all_spectrograms(x_path, lbl_path, out_dir="spectrograms_bin", mode="binary"):
    """
    mode: 'raw' | 'resized' | 'binary'
      - 'raw': оригінальне зображення без змін
      - 'resized': 224x224, але без бінаризації
      - 'binary': 224x224 + бінаризація (default)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Завантаження сигналів і міток
    X = load_signals_from_hdf5(x_path)
    LBL = sio.loadmat(lbl_path)['lbl_train']

    for idx, (signal, meta) in enumerate(tqdm(zip(X, LBL), total=X.shape[0])):
        cls_id = int(meta[0])
        snr = int(meta[1])
        mod_name = modulation_labels.get(cls_id, f"Class{cls_id}")

        # Створення підпапки для класу
        class_dir = os.path.join(out_dir, mod_name)
        os.makedirs(class_dir, exist_ok=True)

        # Побудова спектрограми
        stft_img = generate_stft_image(signal)
        stft_img = np.interp(stft_img, (stft_img.min(), stft_img.max()), (0, 255)).astype(np.uint8)

        if mode == "resized" or mode == "binary":
            stft_img = cv2.resize(stft_img, (224, 224), interpolation=cv2.INTER_CUBIC)

        if mode == "binary":
            _, stft_img = cv2.threshold(stft_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Збереження
        filename = f"{mod_name}_SNR{snr}_{idx:06d}.png"
        filepath = os.path.join(class_dir, filename)
        cv2.imwrite(filepath, stft_img)

    print(f"✅ Збережено {X.shape[0]} спектрограм у '{out_dir}' (режим: {mode})")

# === ВИКЛИК ПРИКЛАД ===
train_path = "/Users/mykola.fedechko/workspace/AI/dataset/DeepRadar2022/X_train.mat"
lbl_path = "/Users/mykola.fedechko/workspace/AI/dataset/DeepRadar2022/lbl_train.mat"
save_all_spectrograms(train_path, lbl_path, mode="binary")