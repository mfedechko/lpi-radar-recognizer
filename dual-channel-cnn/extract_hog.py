import os
import argparse
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

def get_selected_classes(part):
    if part == 1:
        return [modulation_labels[i] for i in range(1, 13)]
    elif part == 2:
        return [modulation_labels[i] for i in range(13, 24)]
    else:
        raise ValueError("Part must be 1 or 2")

def extract_hog_features(image_path):
    image = imread(image_path)
    if image.ndim == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
    features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features

def extract_features_for_part(dataset_path, class_part, output_file):
    selected_classes = get_selected_classes(class_part)
    features, labels, filenames = [], [], []

    for class_name in selected_classes:
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_folder):
            print(f"[!] Warning: {class_folder} does not exist. Skipping.")
            continue

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

    with open(output_file, "wb") as f:
        pickle.dump({
            "features": features,
            "labels": labels,
            "filenames": filenames
        }, f)

    print(f"[✓] HOG features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset folder")
    parser.add_argument("--class-part", type=int, choices=[1, 2], required=True, help="Part of classes to extract (1 or 2)")
    parser.add_argument("--output", default=None, help="Output .pkl file path")
    args = parser.parse_args()

    output = args.output or f"hog_features_part{args.class_part}.pkl"
    extract_features_for_part(args.dataset, args.class_part, output)
