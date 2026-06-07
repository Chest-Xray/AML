import time
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from chest_xray.data.labels import CLASSES

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count the total and trainable parameters of a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    disease_labels: List[str],
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Evaluate a multi-label classification model."""
    model.eval()

    all_probs = []
    all_labels = []

    start_time = time.time()

    # Disable gradient calculation during evaluation.
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            # Model outputs raw logits.
            logits = model(images)

            # Convert logits to probabilities for multi-label classification.
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time

    # Combine all batches into one array.
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    # Convert probabilities to binary predictions.
    y_pred = (y_prob >= threshold).astype(int)

    results = []
    confusion_matrices = {}

    # Evaluate each disease as a separate binary classification task.
    for i, disease in enumerate(disease_labels):
        true_label = y_true[:, i]
        pred_label = y_pred[:, i]
        prob_label = y_prob[:, i]

        # AUROC can fail if only one class is present in y_true.
        try:
            auroc = roc_auc_score(true_label, prob_label)
        except ValueError:
            auroc = np.nan

        precision = precision_score(true_label, pred_label, zero_division=0)
        recall = recall_score(true_label, pred_label, zero_division=0)
        f1 = f1_score(true_label, pred_label, zero_division=0)

        # Matrix order with labels=[0, 1]:
        # [[TN, FP],
        #  [FN, TP]]
        cm = confusion_matrix(true_label, pred_label, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        confusion_matrices[disease] = {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "matrix": cm,
        }

        results.append(
            {
                "Disease": disease,
                "AUROC": auroc,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    results_df = pd.DataFrame(results)

    # Macro metrics are the average across all disease labels.
    macro_auroc = np.nanmean(results_df["AUROC"])
    macro_precision = results_df["Precision"].mean()
    macro_recall = results_df["Recall"].mean()
    macro_f1 = results_df["F1-score"].mean()

    total_params, trainable_params = count_parameters(model)

    summary = {
        "Macro AUROC": macro_auroc,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-score": macro_f1,
        "Inference Time Seconds": inference_time,
        "Inference Time Per Image Seconds": inference_time / len(dataloader.dataset),
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
    }

    return results_df, summary, confusion_matrices

def evaluate_bbox(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    disease_labels: List[str],
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, Any]]]:
    class NoBboxLoader:
        """
        Removes bboxes from the dataloader for classification evaluation
        """
        def __init__(self, loader):
            self.loader = loader
        
        def __iter__(self):
            for images, classification_target, _ in self.loader:
                yield images, classification_target

        def __len__(self) -> int:
            return len(self.loader)
        
        @property
        def dataset(self):
            return self.loader.dataset
    
    class NoBboxModel(torch.nn.Module):
        """
        Remove bbox predictions for classification evaluation
        """
        def __init__(self, bbox_model: torch.nn.Module):
            super().__init__()
            self.bbox_model = bbox_model
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            classification_pred, _ = self.bbox_model(x)
            return classification_pred
        
    classification_results, classification_summary, confusion_matrices = evaluate_model(
        NoBboxModel(model),
        NoBboxLoader(dataloader),
        device,
        list(CLASSES)
    )

    def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return torch.stack([x, y, x + w, y + h], dim=-1)

    def paired_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """IoU between pred[i] and gt[i] for each i. Input shape: (K, 4) xyxy."""
        inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_gt   = (gt_boxes[:, 2]   - gt_boxes[:, 0])   * (gt_boxes[:, 3]   - gt_boxes[:, 1])
        return inter / (area_pred + area_gt - inter).clamp(min=1e-6)

    iou_per_disease: Dict[str, List[float]] = {label: [] for label in disease_labels}

    model.eval()
    with torch.no_grad():
        for images, classification_target, bbox_target in dataloader:
            images = images.to(device)
            classification_target = classification_target.to(device)
            bbox_target = bbox_target.to(device)

            _, bbox_pred = model(images)

            # We only evaluate if the disease is present and a bbox was annotated
            disease_present = classification_target.bool()
            bbox_annotated = (bbox_target[..., 2] > 0) & (bbox_target[..., 3] > 0)
            valid_mask = disease_present & bbox_annotated

            for c, label in enumerate(disease_labels):
                sample_mask = valid_mask[:, c]
                if not sample_mask.any():
                    continue
                pred_xyxy = xywh_to_xyxy(bbox_pred[sample_mask, c])
                gt_xyxy   = xywh_to_xyxy(bbox_target[sample_mask, c,])
                iou_per_disease[label].extend(paired_iou(pred_xyxy, gt_xyxy).cpu().tolist())

    bbox_summary: Dict[str, Any] = {}
    per_disease_ious = []
    for label in disease_labels:
        scores = iou_per_disease[label]
        if scores:
            mean_iou = float(sum(scores) / len(scores))
            bbox_summary[label] = {"mean_iou": mean_iou, "n_samples": len(scores)}
            per_disease_ious.append(mean_iou)
        else:
            bbox_summary[label] = {"mean_iou": float("nan"), "n_samples": 0}

    bbox_summary["mIoU"] = float(sum(per_disease_ious) / len(per_disease_ious)) if per_disease_ious else float("nan")

    return classification_results, classification_summary, confusion_matrices, bbox_summary
