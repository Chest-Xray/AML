from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
from chest_xray.models.bboxbase import XrayBboxBase
from torch import load
from pathlib import Path
import pickle

CLASSIFIER_EPOCHS = 5
REGRESSOR_EPOCHS = 25

def main():
    # train classifier
    classifier = XrayClassifierBase(type="vgg16")
    train_transform = classifier.modelTrainer.trainsform_train(classifier.modelTrainer.image_size)
    test_transform = classifier.modelTrainer.transform_images(classifier.modelTrainer.image_size)
    train_loader, test_loader = classifier.modelTrainer.cv.test_loaders(train_transform, test_transform)

    path = classifier.trainModel_no_cv(train_loader, test_loader, CLASSIFIER_EPOCHS)
    # train regressor
    regressor = XrayBboxBase("vgg16", model=load(path, weights_only=False))
    bbox_path = regressor.trainModel_no_cv(train_loader, test_loader, REGRESSOR_EPOCHS)
    results_df, summary, confusion_matrices, bbox_summary = regressor.evaluate(test_loader)
    
    # save evaluations as pickles
    pickle_path = Path(__file__) / "chest_xray" / "data" / "pickles"
    pickle.dump(results_df, open(pickle_path / "results_df.pkl", "wb"))
    pickle.dump(summary, open(pickle_path / "summary.pkl", "wb"))
    pickle.dump(confusion_matrices, open(pickle_path / "confusion_matrices.pkl", "wb"))
    pickle.dump(bbox_summary, open(pickle_path / "bbox_summary.pkl", "wb"))


if __name__ == "__main__":
    main()