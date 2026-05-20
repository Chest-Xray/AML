import torch
import torchvision.models as models
import wandb
from collections.abc import Callable, Iterable
from timm.loss import AsymmetricLossMultiLabel
from typing import Literal

from chest_xray.models.train import ModelTrainer
from chest_xray.data.labels import CLASSES

 
BATCH_SIZE: int = 4    # play around with this on Habrok
SEED: int = 42
NUM_WORKERS: int = 4    # play around with this on Habrok
K_FOLDS: int = 4


class XrayClassifierBase(torch.nn.Module):
    def __init__(
            self: XrayClassifierBase,
            type: Literal["vgg16", "densenet"] = "vgg16",
            criterion: Callable[[tuple[torch.Tensor, torch.Tensor]],float] = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05),
            pretrained: bool = True,
            optimizer = torch.optim.Adam,
            lr: float = 0.001
            ) -> None:
        self.type = type
        self.model = None
        self._build_model()
        self.optimizer = self.optimizer(self.model.parameters(), lr=0.001)
        self.pretrained = pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.modelTrainer = ModelTrainer(self.model, self.criterion, self.optimizer, self.device)


    def _build_model(self):
        if self.type == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights)
        elif self.type == "densenet":
            if self.pretrained:
                self.model = models.densenet161(weights=models.DenseNet161_Weights)
            else:
                self.model = models.densenet161(weights=None)
        else:
            raise ValueError("Please select a model type.")

        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(CLASSES))

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



    def trainModel(self):
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)
    
        for fold_idx, (train_loader, val_loader) in enumerate(
            self.modelTrainer.yield_dataloaders(transform)
        ):
            print(f"\nFold {fold_idx + 1}/{K_FOLDS}")
    
            run = wandb.init(
                project="chest_xray",
                entity="chest_xray",
                name=f"fold_{fold_idx+1}",
                config={
                    "fold": fold_idx + 1,
                    "epochs": 10,
                    "batch_size": BATCH_SIZE,
                    "lr": 1e-3,
                    "model": self.type,
                    "pretrained": self.pretrained
                },
            )
    
            for epoch in range(10):
                train_loss = self.modelTrainer.train_one_epoch(
                    epoch, 10, f"Fold {fold_idx+1}", train_loader
                )
                val_loss = self.modelTrainer.validate_one_epoch(
                    epoch, 10, f"Fold {fold_idx+1}", val_loader
                )
    
                # log to wandb
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
    
                print(
                    f"Fold {fold_idx+1} Epoch {epoch+1}: "
                    f"Train {train_loss:.4f} | Val {val_loss:.4f}"
                )
    
            run.finish()
    