from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
import sys

def main():
    model: str = sys.argv[1]
    if model == "baseline":
        classifier = XrayClassifierBase(type="vgg16")
    elif model == "pretrained":
        classifier = XrayClassifierBase(type="densenet", pretrained=True)
    elif model == "scratch":
        classifier = XrayClassifierBase(type="densenet", pretrained=False)
    else:
        raise Exception(f"{model} is not one of the models, use baseline, pretrained, or scratch")
    classifier.trainModel()

if __name__ == "__main__":
    main()
