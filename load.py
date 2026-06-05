
from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
from chest_xray.tools.globals import *
import os
from pathlib import Path
import sys
from torch import load


def load_model() -> XrayClassifierBase:
    modelname = sys.argv[1]
    path = MODEL_PATH / modelname
    if not os.path.exists(path):
        raise FileNotFoundError(f"path {path} not found")
    if "densenet161" in path and "pretrained" in path:
        classifier = XrayClassifierBase("densenet161", pretrained=True, model=load(path, weights_only=False))
    if "densenet161" in path and "scratch" in path:
        classifier = XrayClassifierBase("densenet161", pretrained=False, model=load(path, weights_only=False))
    if "densenet201" in path and "pretrained" in path:
        classifier = XrayClassifierBase("densenet201", pretrained=True, model=load(path, weights_only=False))
    if "densenet201" in path and "scratch" in path:
        classifier = XrayClassifierBase("densenet201", pretrained=False, model=load(path, weights_only=False))
    if "vgg16" in path:
        classifier = XrayClassifierBase("vgg16", model=load(path, weights_only=False))
    return classifier



if __name__ == "__main__":
    model = load_model()
    model.evaluate()
