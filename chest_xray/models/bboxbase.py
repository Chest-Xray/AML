import torch
import torch.nn.functional as F
import torchvision.ops as ops
from timm.loss import AsymmetricLossMultiLabel
from typing import Literal
from chest_xray.data.labels import CLASSES
from tqdm import tqdm
from ..tools.globals import *
from ..features.evaluation import evaluate_bbox
from .xraymodelbaseclass import XrayClassifierBase
from .train import ModelTrainer
from torch import load


class BboxDenseNet(torch.nn.Module):
    """
    Adds a bounding box regression head to the DenseNet trained on classification only
    """

    def __init__(self, backbone: torch.nn.Module, num_classes: int) -> None:
        super().__init__()
        in_features: int = backbone.classifier.in_features
        self.features = backbone.features
        self.classifier = backbone.classifier          # Linear(in_features, num_classes)
        self.bbox_head = torch.nn.Linear(in_features, num_classes * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat = F.relu(feat, inplace=True)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)                               # (B, in_features)
        cls_logits = self.classifier(feat)                          # (B, C)
        bbox_pred  = self.bbox_head(feat).view(feat.size(0), -1, 4) # (B, C, 4)
        return cls_logits, bbox_pred


class XrayBboxBase(XrayClassifierBase):
    """
    Extends XrayClassifierBase with a parallel bounding box regression head.
    Using the BboxDenseNet above, we train using both GIoU for bbox regression
    and the same Asymmetric loss for classification.
    """
    def __init__(
        self,
        type: Literal["densenet161", "densenet201"] = "densenet201",
        classification_criterion=AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05),
        pretrained: bool = True,
        optimizer=torch.optim.Adam,
        lr: float = 0.001,
        model=None,
        bbox_loss_weight: float = 2.0,
    ) -> None:
        self.bbox_loss_weight = bbox_loss_weight
        self.classification_optimizer = optimizer  # save class before super() replaces it with an instance
        super().__init__(
            type=type,
            criterion=classification_criterion,
            pretrained=pretrained,
            optimizer=optimizer,
            lr=lr,
            model=model,
        )
        self.classification_criterion = self.criterion
    
        if not isinstance(self.model, BboxDenseNet):
            self.model = BboxDenseNet(self.model, len(CLASSES)).to(self.device)
            self.optimizer = self.classification_optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
            )
            self.modelTrainer = ModelTrainer(
                self.model, self.criterion, self.optimizer, self.device
            )

    def _build_model(self) -> None:
        if self.type not in ("densenet161", "densenet201"):
            raise ValueError(
                f"XrayBboxBase does not support type='{self.type}'. "
                "Choose 'densenet161' or 'densenet201'."
            )
        super()._build_model()
        self.model = BboxDenseNet(self.model, len(CLASSES)).to(self.device)

    
    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [..., (x, y, w, h)] → [..., (x1, y1, x2, y2)]."""
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return torch.stack([x, y, x + w, y + h], dim=-1)


    def _compute_loss(
        self,
        classification_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        classification_target: torch.Tensor,
        bbox_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute both asymmetric loss, and Generalized Intersection over Union loss
        """
        classification_loss = self.classification_criterion(classification_pred, classification_target)

        # We mask the bbox loss to only count when both a disease and corresponding bounding box are present
        disease_present = classification_target.bool()
        bbox_annotated  = (bbox_target[..., 2] > 0) & (bbox_target[..., 3] > 0)
        valid_mask      = disease_present & bbox_annotated

        if valid_mask.any():
            print("VALID MASK FOUND")
            pred_xyxy = self._xywh_to_xyxy(bbox_pred[valid_mask])
            tgt_xyxy  = self._xywh_to_xyxy(bbox_target[valid_mask])
            bbox_loss = ops.generalized_box_iou_loss(
                pred_xyxy, tgt_xyxy, reduction="mean"
            )
        else:
            n_disease  = disease_present.sum().item()
            n_bbox     = bbox_annotated.sum().item()
            n_valid    = valid_mask.sum().item()
            print(
                f"[bbox_loss=0] disease_present={n_disease} | "
                f"bbox_annotated={n_bbox} | "
                f"valid={n_valid}"
            )
            print(f"  classification_target sample: {classification_target[0]}")
            print(f"  bbox_target sample w/h:       {bbox_target[0, :, 2:4]}")
            bbox_loss = torch.tensor(0.0, device=self.device)

        total_loss = classification_loss + self.bbox_loss_weight * bbox_loss
        return total_loss, classification_loss, bbox_loss


    def _run_epoch(self, loader, train: bool, epoch: int) -> tuple[float, float, float]:
        """Run one train or validation epoch. Returns total, classification, and bbox losses."""
        self.model.train(train)
        total_sum = classification_sum = bbox_sum = 0.0
        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch [{epoch}/{NUM_EPOCHS}] {'train' if train else 'val'}",
            unit="batch",
            dynamic_ncols=True
        )

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch_idx, (images, classification_target, bbox_target) in enumerate(progress_bar, start=1):
                images = images.to(self.device)
                classification_target = classification_target.to(self.device)
                bbox_target = bbox_target.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                classifiction_pred, bbox_pred = self.model(images)
                loss, c_loss, b_loss  = self._compute_loss(
                    classifiction_pred, bbox_pred, classification_target, bbox_target
                )

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_sum += loss.item()
                classification_sum   += c_loss.item()
                bbox_sum  += b_loss.item()

                progress_bar.set_postfix(
                    loss=f"{total_sum / batch_idx:.4f}",
                    c_loss=f"{classification_sum / batch_idx:.4f}",
                    b_loss=f"{bbox_sum / batch_idx:.4f}" 
                )

        n = len(loader)
        return total_sum / n, classification_sum / n, bbox_sum / n


    def trainModel(self) -> None:
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)

        for fold_idx, (train_loader, val_loader) in enumerate(
            self.modelTrainer.yield_dataloaders(transform)
        ):
            print(f"\nFold {fold_idx + 1}/{K_FOLDS}")

            for epoch in range(NUM_EPOCHS):
                train_loss_total, train_loss_classification, train_loss_bbox = self._run_epoch(train_loader, train=True, epoch=epoch+1)
                val_loss_total, val_loss_classification, val_loss_bbox = self._run_epoch(val_loader,   train=False, epoch=epoch+1)

                print(
                    f"Fold {fold_idx+1} Epoch {epoch+1}: "
                    f"Train loss {train_loss_total:.4f} (classification {train_loss_classification:.4f}, bbox {train_loss_bbox:.4f})"
                    f"Val loss {val_loss_total:.4f} (classification {val_loss_classification:.4f}, bbox {val_loss_bbox:.4f})"
                )

                path = (
                    f"{MODEL_PATH}{self.type}_bbox_"
                    f"{'pretrained' if self.pretrained else 'scratch'}_epoch{epoch}.pth"
                )
                torch.save(self.model, path)
                print(f"Model saved: {path}")

                results_df, summary, confusion_matrices, bbox_summary = self.evaluate(val_loader)

    

    def trainModel_no_cv(self, train_loader, val_loader, max_epochs) -> None:
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)

        path = ''
        train_losses = []
        val_losses = []
        paths = []
        early_stop: bool = False
        for epoch in range(max_epochs):
            train_loss_total, train_loss_classification, train_loss_bbox = self._run_epoch(train_loader, train=True, epoch=epoch+1)
            val_loss_total, val_loss_classification, val_loss_bbox = self._run_epoch(val_loader,   train=False, epoch=epoch+1)
            train_losses.append(train_loss_total)
            val_losses.append(val_loss_total)
            print(
                f"Epoch {epoch+1}: "
                f"Train loss {train_loss_total:.4f} (classification {train_loss_classification:.4f}, bbox {train_loss_bbox:.4f})"
                f"Val loss {val_loss_total:.4f} (classification {val_loss_classification:.4f}, bbox {val_loss_bbox:.4f})"
            )
            path = (
                f"{MODEL_PATH}{self.type}_bbox_"
                f"{'pretrained' if self.pretrained else 'scratch'}_epoch{epoch}.pth"
            )
            paths.append(path)
            torch.save(self.model, path)
            print(f"Model saved: {path}")
            if len(train_losses) > 2:
                if train_losses[-1] > train_losses[-2] and train_losses[-2] > train_losses[-3]:
                    early_stop = True
                    break
                if val_losses[-1] > val_losses[-2] and val_losses[-2] > val_losses[-3]:
                    early_stop = True
                    break
        if early_stop:
            return paths[-3]
        else:
            return path


    def evaluate(self, eval_loader):
        results_df, summary, confusion_matrices, bbox_summary = evaluate_bbox(
            self.model,
            eval_loader,
            self.device,
            list(CLASSES),
        )
        print("Summary:")
        for key, val in summary.items():
            print(f"  {key}: {val}")
        print("\nDataframe:")
        print(results_df)
        print("\nConfusion matrices:")
        print(confusion_matrices)
        print("\nbbox summary")
        print(bbox_summary)
        return results_df, summary, confusion_matrices, bbox_summary

if __name__ == "__main__":
    path = MODEL_PATH / "densenet_pretrained_epoch10.pth"
    bbox = XrayBboxBase("densenet161", model=load(path, weights_only=False))
    bbox.trainModel()
    
