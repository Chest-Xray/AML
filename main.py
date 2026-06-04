from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
import sys

def main():
    model: str = sys.argv[1]
    if model == "vgg16":
        classifier = XrayClassifierBase(type="vgg16")
    elif model == "densenet161":
        if sys.argv[2] == "pretrained":
            classifier = XrayClassifierBase(type="densenet161", pretrained=True)
        elif sys.argv[2] == "scratch":
            classifier = XrayClassifierBase(type="densenet161", pretrained=False)
        else:
            raise Exception(f"{sys.argv[2]} is not one of the options for model {model}, choose scratch or pretrained")
    elif model == "densenet201":
        if sys.argv[2] == "pretrained":
            classifier = XrayClassifierBase(type="densenet201", pretrained=True)
        elif sys.argv[2] == "scratch":
            classifier = XrayClassifierBase(type="densenet201", pretrained=False)
        else:
            raise Exception(f"{sys.argv[2]} is not one of the options for model {model}, choose scratch or pretrained")
    else:
        raise Exception(f"{model} is not one of the models, use baseline, pretrained, or scratch")
    classifier.trainModel()

if __name__ == "__main__":
    main()
