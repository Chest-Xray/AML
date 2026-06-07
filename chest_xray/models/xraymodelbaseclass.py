import torch
import torchvision.models as models
from collections.abc import Callable, Iterable
from timm.loss import AsymmetricLossMultiLabel
from typing import Literal
from pathlib import Path
from chest_xray.models.train import ModelTrainer
from chest_xray.data.labels import CLASSES
from ..tools.globals import *
from ..features.evaluation import evaluate_model


class XrayClassifierBase(torch.nn.Module):
    def __init__(
            self,
            type: Literal["vgg16", "densenet161", "densenet201"] = "vgg16",
            criterion: Callable[[tuple[torch.Tensor, torch.Tensor]],float] = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05),
            pretrained: bool = True,
            optimizer = torch.optim.Adam,
            lr: float = 0.001,
            # model: torch.nn.module | None = None
            model = None
            ) -> None:
        super().__init__()
        self.type = type
        self.model = model
        self.pretrained = pretrained
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None:
            self._build_model()
        self.optimizer = optimizer
        self.optimizer = self.optimizer(filter(lambda parameter: parameter.requires_grad, self.model.parameters()), lr=self.lr)
        self.criterion = criterion
        self.modelTrainer = ModelTrainer(self.model, self.criterion, self.optimizer, self.device)


    def _build_model(self):
        if self.type == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.model.classifier[6] = torch.nn.Linear(
                self.model.classifier[6].in_features,
                len(CLASSES)
            )
        elif self.type == "densenet161":
            if self.pretrained:
                self.model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
            else:
                self.model = models.densenet161(weights=None)
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(CLASSES))
        elif self.type == "densenet201":
            if self.pretrained:
                self.model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            else:
                self.model = models.densenet201(weights=None)
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(CLASSES))
        else:
            raise ValueError("Please select a model type.")


        # take first layer
        old_first_layer = self.model.features[0]

        # Create new convolution layer with 1 input channel and keep the rest the same
        new_first_layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=old_first_layer.out_channels,
            kernel_size=old_first_layer.kernel_size,
            stride=old_first_layer.stride,
            padding=old_first_layer.padding,
        )

        # Initialize the new first layer's weights by averaging the weights of the original 3 channels
        with torch.no_grad():
            new_first_layer.weight = torch.nn.Parameter(
                old_first_layer.weight.mean(dim=1, keepdim=True)
            )
            new_first_layer.bias = old_first_layer.bias

        self.model.features[0] = new_first_layer
        self.model = self.model.to(self.device)


    def _log(self, train_loss, val_loss, results_df, summary, confusion_matrices, fold_idx, epoch):
        logs_path: Path = Path(__file__).parent.parent.parent / "data" / "logs"
        log_name = f"{self.type}" if self.type == "vgg16" else f"{self.type}_{'pretrained' if self.pretrained else 'scratch'}"
        log_name = log_name + f"_lr_{self.lr}"
        log_name = log_name + f"_fold_{fold_idx}.txt"
        log_path = logs_path / log_name
        if not log_path.exists():
            with open(log_path, 'w') as file:
                file.write("")
        with open(log_path, "a") as file:
            file.write(f"Epoch {epoch}:\n")
            file.write(f"Train {train_loss:.4f} | Val {val_loss:.4f}\n")
            file.write("Summary:\n")
            for key, val in summary.items():
                file.write(f"  {key}: {val}\n")
            file.write("\ndataframe:\n")
            file.write(results_df)
            file.write("\n\nconfusion matrices:\n")
            file.write(confusion_matrices)
            file.write("\n\n\n")


    def trainModel(self):
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)

        for fold_idx, (train_loader, val_loader) in enumerate(
            self.modelTrainer.yield_dataloaders(transform)
        ):
            print(f"\nFold {fold_idx + 1}/{K_FOLDS}")
    
            for epoch in range(NUM_EPOCHS):
                train_loss = self.modelTrainer.train_one_epoch(
                    epoch, NUM_EPOCHS, f"Fold {fold_idx+1}", train_loader
                )
                val_loss = self.modelTrainer.validate_one_epoch(
                    epoch, NUM_EPOCHS, f"Fold {fold_idx+1}", val_loader
                )
    
                print(
                    f"Fold {fold_idx+1} Epoch {epoch+1}: "
                    f"Train {train_loss:.4f} | Val {val_loss:.4f}"
                )
                path: str = f"{MODEL_PATH}{self.type}_{'pretrained' if self.pretrained else 'scratch'}_epoch{epoch}.pth"
                torch.save(self.model, path)
                print(f"model saved: {path}")
                results_df, summary, confusion_matrices = self.evaluate(val_loader)
                self._log(train_loss, val_loss, results_df, summary, confusion_matrices, fold_idx + 1, epoch)


    
    def evaluate(self, eval_loader):
        classes = [c for c in CLASSES]
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)
        results_df, summary, confusion_matrices = evaluate_model(
            self.model,
            eval_loader,
            self.device,
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

