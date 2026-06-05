import os

import torch
from torch import load

from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
from chest_xray.tools.globals import MODEL_PATH
from chest_xray.models.train import CLASSES


class BoundingBoxRegressionModel:
    def __init__(self, model_name):
        """Instantiate a BoundingBoxRegressionModel from an XrayClassifier model"""
        path = MODEL_PATH / model_name
        if not os.path.exists(path):
            raise FileNotFoundError(f"path {path} not found")
        if "densenet161" in path and "pretrained" in path:
            self.model = XrayClassifierBase("densenet161", pretrained=True, model=load(path, weights_only=False))
        if "densenet161" in path and "scratch" in path:
            self.model = XrayClassifierBase("densenet161", pretrained=False, model=load(path, weights_only=False))
        if "densenet201" in path and "pretrained" in path:
            self.model = XrayClassifierBase("densenet201", pretrained=True, model=load(path, weights_only=False))
        if "densenet201" in path and "scratch" in path:
            self.model = XrayClassifierBase("densenet201", pretrained=False, model=load(path, weights_only=False))
        if "vgg16" in path:
            self.model = XrayClassifierBase("vgg16", model=load(path, weights_only=False))
        self._adjust_output()
        
    
    def _adjust_output(self):
        """Adjust the output of the classifier to predict 75 classes (15 classes, 5 features per class)"""
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(CLASSES)*5)
        self.model = self.model.to(self.model.device)


if __name__ == "__main__":
    pass