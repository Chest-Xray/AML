from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
from chest_xray.models.bboxbase import XrayBboxBase
from chest_xray.tools.globals import *
from chest_xray.data.labels import CLASSES
from chest_xray.features.evaluation import evaluate_bbox

import sys
from torch import load


def load_model() -> XrayClassifierBase:
    path = sys.argv[1]
    if "vgg16" in path:
        classifier = XrayBboxBase("vgg16", model=load(path, weights_only=False))
    if "densenet201" in path and "pretrained" in path:
        classifier = XrayBboxBase("densenet201", pretrained=True, model=load(path, weights_only=False))
    train_transform = classifier.modelTrainer.trainsform_train(classifier.modelTrainer.image_size)
    test_transform = classifier.modelTrainer.transform_images(classifier.modelTrainer.image_size)
    evaluate_bbox(
        classifier.model,
        classifier.modelTrainer.cv.test_loaders(train_transform, test_transform)[1],
        classifier.device,
        disease_labels=CLASSES
    )


if __name__ == "__main__":
    model = load_model()
