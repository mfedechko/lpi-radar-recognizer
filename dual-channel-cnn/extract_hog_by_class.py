
import os
import pickle
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm

modulation_labels = {
    1: "LFM", 2: "2FSK", 3: "4FSK", 4: "8FSK", 5: "Costas",
    6: "2PSK", 7: "4PSK", 8: "8PSK", 9: "Barker", 10: "Huffman",
    11: "Frank", 12: "P1", 13: "P2", 14: "P3", 15: "P4", 16: "Px",
    17: "ZadoffChu", 18: "T1", 19: "T2", 20: "T3", 21: "T4",
    22: "NM", 23: "Noise"
}

# 🧠 Можеш змінити тут список класів, якщо хочеш обробити тільки частину
selected_classes = [
    "LFM", "4FSK", "Costas", "2PSK", "Barker", "Huffman", "Frank",
    "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"
]

def extract_hog_features(image_path):
    image = imread(image_path)
    if image.ndim == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
    features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features

def extract_features_by_class(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for class_name in selected_classes:
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_folder):
            print(f"[!] Warning: {class_folder} does not exist. Skipping.")
            continue

        features, labels, filenames = [], [], []

        for file in tqdm(os.listdir(class_folder), desc=f"Processing {class_name}"):
            if file.endswith(".png") or file.endswith(".jpg"):
                image_path = os.path.join(class_folder, file)
                try:
                    hog_feat = extract_hog_features(image_path)
                    features.append(hog_feat)
                    labels.append(class_name)
                    filenames.append(file)
                except Exception as e:
                    print(f"[x] Error processing {image_path}: {e}")

        output_file = os.path.join(output_dir, f"{class_name}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump({
                "features": features,
                "labels": labels,
                "filenames": filenames
            }, f)
        print(f"✅ Saved {len(features)} samples to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to spectrogram image dataset")
    parser.add_argument("--output", type=str, default="hog_features", help="Where to save class-wise pkl files")
    args = parser.parse_args()

    extract_features_by_class(args.input, args.output)
