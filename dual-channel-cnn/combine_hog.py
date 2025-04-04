import pickle

with open("hog_features_part1.pkl", "rb") as f1, open("hog_features_part2.pkl", "rb") as f2:
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)

# Об'єднуємо
combined = {
    "features": data1["features"] + data2["features"],
    "labels": data1["labels"] + data2["labels"],
    "filenames": data1["filenames"] + data2["filenames"]
}

# Зберігаємо в один файл
with open("hog_features_all.pkl", "wb") as f:
    pickle.dump(combined, f)