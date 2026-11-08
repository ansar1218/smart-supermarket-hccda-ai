"""Retrain the grocery recognizer from scratch.

Lab 4 equivalent: training the model that "Model Management" registers.
Usage: python train.py          (downloads the dataset if needed, ~250 MB, ~5-10 min on CPU)
"""
import os
import subprocess

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model import DEFAULT_MODEL, GroceryRecognizer, ROOT

DATA_ROOT = os.path.join(ROOT, "GroceryStoreDataset")
DATA_DIR = os.path.join(DATA_ROOT, "dataset")


def load_split(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.txt"), header=None, names=["path", "class_id", "coarse_id"])
    df["path"] = DATA_DIR + "/" + df.path.str.strip()
    df["split"] = name
    return df


def main():
    if not os.path.isdir(DATA_ROOT):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/marcusklasson/GroceryStoreDataset.git", DATA_ROOT], check=True)
    df = pd.concat([load_split(s) for s in ("train", "val", "test")], ignore_index=True)
    classes = pd.read_csv(os.path.join(DATA_DIR, "classes.csv"))
    names = dict(zip(classes.iloc[:, 1], classes.iloc[:, 0]))
    coarse = dict(zip(classes.iloc[:, 1], classes.iloc[:, 3]))

    # Same feature extractor the app uses, so training and serving match exactly
    extractor = GroceryRecognizer(model_path=None)

    print(f"Extracting features for {len(df)} images...")
    feats = []
    for i in range(0, len(df), 64):
        feats.append(extractor.features([Image.open(p) for p in df.path[i:i + 64]]))
    F = np.concatenate(feats)
    y = df.class_id.values
    tr, va, te = [(df.split == s).values for s in ("train", "val", "test")]

    scaler = StandardScaler().fit(F[tr])
    best = None
    for C in np.logspace(-3, 0, 4):
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(F[tr]), y[tr])
        acc = clf.score(scaler.transform(F[va]), y[va])
        print(f"C={C:<6.3g} val accuracy {acc:.3f}")
        if best is None or acc > best[1]:
            best = (C, acc, clf)
    clf = best[2]

    pred = clf.predict(scaler.transform(F[te]))
    to_coarse = np.vectorize(coarse.get)
    print(f"Test: exact {np.mean(pred == y[te]):.1%} | type {np.mean(to_coarse(pred) == to_coarse(y[te])):.1%}")

    os.makedirs(os.path.dirname(DEFAULT_MODEL), exist_ok=True)
    joblib.dump({"scaler": scaler, "classifier": clf, "class_names": names}, DEFAULT_MODEL)
    print("Saved", DEFAULT_MODEL)


if __name__ == "__main__":
    main()
