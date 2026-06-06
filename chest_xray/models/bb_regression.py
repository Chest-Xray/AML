import os

import torch
from torch import load

from ..tools.globals import MODEL_PATH, NUM_EPOCHS, K_FOLDS
from ..models.xraymodelbaseclass import XrayClassifierBase
from ..data.labels import CLASSES
from ..features.evaluation import evaluate_model

class BoundingBoxRegression:
    """Instantiate a BoundingBoxRegressionModel from an XrayClassifier model"""
    def __init__(self, model_name):
        path = MODEL_PATH / model_name
        if not os.path.exists(path):
            raise FileNotFoundError(f"path {path} not found")
        path = str(path)

        if "densenet161" in path and "pretrained" in path:
            self.classifier = XrayClassifierBase("densenet161", pretrained=True, model=load(path, map_location = torch.device('cpu'), weights_only=False))
        if "densenet161" in path and "scratch" in path:
            self.classifier XrayClassifierBase("densenet161", pretrained=False, model=load(path, map_location = torch.device('cpu'), weights_only=False))
        if "densenet201" in path and "pretrained" in path:
            self.classifier XrayClassifierBase("densenet201", pretrained=True, model=load(path, map_location = torch.device('cpu'), weights_only=False))
        if "densenet201" in path and "scratch" in path:
            self.classifier XrayClassifierBase("densenet201", pretrained=False, model=load(path, map_location = torch.device('cpu'), weights_only=False))
        if "vgg16" in path:
            self.classifier XrayClassifierBase("vgg16", model=load(path, map_location = torch.device('cpu'), weights_only=False))


    def adjust_output(self, model):
        """Adjust the output of the classifier to predict 75 classes (15 classes, 5 features per class)"""
        self.classifier.model.classifier = torch.nn.Linear(model.model.classifier.in_features, len(CLASSES)*5)
        model = self.classifier.model.to(self.classifier.model.device)


    def trainModel(self):
        transform = self.classifier.modelTrainer.transform_images(self.classifier.modelTrainer.image_size)

        # Freeze gradients for every layer
        for name, parameter in self.classifier.model.named_parameters():
            try:
                parameter.requires_grad = False
            except:
                if "model.classifier" in name:
                    pass
        
        for fold_idx, (train_loader, val_loader) in enumerate(
            self.classifier.modelTrainer.yield_dataloaders(transform)
        ):
            print(f"\nFold {fold_idx + 1}/{K_FOLDS}")
            unfreezed_idx = 2
            running_idx = 0
            for epoch in range(NUM_EPOCHS):
                if epoch > 0:
                    for name, parameter in reversed(self.classifier.model.named_parameters()):
                        if running_idx < 3 and parameter.requires_grad == False:
                            parameter.requires_grad = True
                            running_idx += 1

                train_loss = self.classifier.modelTrainer.train_one_epoch(
                    epoch, NUM_EPOCHS, f"Fold {fold_idx+1}", train_loader
                )
                val_loss = self.classifier.modelTrainer.validate_one_epoch(
                    epoch, NUM_EPOCHS, f"Fold {fold_idx+1}", val_loader
                )

                print(
                    f"Fold {fold_idx+1} Epoch {epoch+1}: "
                    f"Train {train_loss:.4f} | Val {val_loss:.4f}"
                )
                path: str = f"{MODEL_PATH}{self.classifier.type}_{'pretrained' if self.classifier.pretrained else 'scratch'}_epoch{epoch}.pth"
                torch.save(self.classifier, path)
                print(f"model saved: {path}")
                results_df, summary, confusion_matrices = self.evaluate(val_loader)
                self.classifier._log(train_loss, val_loss, results_df, summary, confusion_matrices, fold_idx + 1, epoch)


        
    def evaluate(self, eval_loader):
        classes = [c for c in CLASSES]
        transform = self.classifier.modelTrainer.transform_images(self.classifier.modelTrainer.image_size)
        results_df, summary, confusion_matrices = evaluate_model(
            self.classifier.model,
            eval_loader,
            self.classifier.device,
            classes
        )
        print("summary:")
        for key, val in summary.items():
            print(f"{key}: {val}")
        print("\n\ndataframe:")
        print(results_df)
        print("\n\nconfusion matrices:")
        print(confusion_matrices)
        return results_df, summary, confusion_matrices


        """
        4 denseblocks

        block 1 
        - 6 layers
            transition
        block 2
        - 12 layers
            transition
        block 3
        - 36 layers
            transition
        block 4
        - 24 layers
        """
        # # Freeze every parameter:
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # # unfreeze layer by layer
        # for layer in range(NUM_EPOCHS):


def main(model_name):
    """Instantiate a BoundingBoxRegressionModel from an XrayClassifier model"""
    model = build_model(model_name)
    adjust_output(model)
    trainModel(model)


if __name__ == "__main__":
    main("densenet161_pretrained_epoch10.pth")