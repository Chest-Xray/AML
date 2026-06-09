from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np

def get_gradcam_bbox(model, input_tensor, class_idx, target_layer, image_np):
    """
    Returns (x, y, w, h) pixel coords from GradCAM heatmap.
    target_layer: last conv layer before your classifier head, e.g. model.features[-1]
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  

    threshold = 0.4
    binary = (grayscale_cam >= threshold).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    scale_x = image_np.shape[1] / grayscale_cam.shape[1]
    scale_y = image_np.shape[0] / grayscale_cam.shape[0]
    return (x * scale_x, y * scale_y, w * scale_x, h * scale_y)