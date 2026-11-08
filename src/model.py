"""Grocery product recognizer: MobileNetV2 features + logistic regression.

Lab 4 equivalent: this is the "AI application" (Model Management) that the
real-time service loads.
"""
import os
import time

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "grocery_classifier.joblib")


def pretty(name):
    return name.replace("-", " ")


class GroceryRecognizer:
    def __init__(self, model_path=DEFAULT_MODEL):
        """model_path=None loads only the feature extractor (used by train.py)."""
        if model_path is not None:
            bundle = joblib.load(model_path)
            self.scaler = bundle["scaler"]
            self.clf = bundle["classifier"]
            self.names = bundle["class_names"]
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v2(weights=weights, progress=False)
        self.backbone.classifier = torch.nn.Identity()
        self.backbone.eval()
        self.prep = weights.transforms()

    def features(self, images):
        batch = torch.stack([self.prep(im.convert("RGB")) for im in images])
        with torch.no_grad():
            return self.backbone(batch).numpy()

    def predict(self, image, top_k=5):
        """Return (list of {product, confidence}, latency in ms) for one PIL image."""
        t0 = time.time()
        proba = self.clf.predict_proba(self.scaler.transform(self.features([image])))[0]
        order = np.argsort(proba)[::-1][:top_k]
        classes = self.clf.classes_
        results = [{"product": pretty(self.names[int(classes[i])]), "confidence": float(proba[i])}
                   for i in order]
        return results, (time.time() - t0) * 1000


if __name__ == "__main__":
    import sys
    rec = GroceryRecognizer()
    for path in sys.argv[1:] or [os.path.join(ROOT, "samples", "lemon.jpg")]:
        res, ms = rec.predict(Image.open(path))
        print(f"{os.path.basename(path)} -> {res[0]['product']} ({res[0]['confidence']:.0%}) in {ms:.0f} ms")
