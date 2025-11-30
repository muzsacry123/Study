#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SAVE_DIR = "E:\\Github\\Study\\python\\machine_learning\\HW\\special\\"
DATA_DIR = "E:\\Github\\Study\\python\\machine_learning\\HW\\special\\data\\train" # 我的路径

def load_data():
    X, y = [], []
    for fn in os.listdir(DATA_DIR):
        if not fn.endswith(".csv"):
            continue
        try:
            label = int(fn.split("-")[-1].split(".")[0])
            arr = np.loadtxt(os.path.join(DATA_DIR, fn))
            X.append(arr)
            y.append(label)
        except Exception as e:
            print(f"[Error] Failed: {fn} => {e}")
    return np.array(X), np.array(y)

def extract_psd_features(x):
    f, Pxx = welch(x, nperseg=512)
    return Pxx[:256]     # 使用前 256 维

def main():
    print("Loading data...")
    X_raw, y = load_data()

    print("Extracting PSD features...")
    X = np.array([extract_psd_features(x) for x in X_raw])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training SVM...")
    model = SVC(kernel="rbf", C=3, gamma="scale", probability=True)
    model.fit(X_train, y_train)

    print("Validating...")
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    print("Validation accuracy:", acc)

    # 保存模型
    joblib.dump(model, SAVE_DIR + "model1_svm.pkl")
    print("Saved model1_svm.pkl")

    # 绘图
    plt.figure(figsize=(6,4))
    plt.plot([acc], marker='o')
    plt.title("Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig(SAVE_DIR + "train1_accuracy.png")
    print("Saved train1_accuracy.png")

if __name__ == "__main__":
    main()
