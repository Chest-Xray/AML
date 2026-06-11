from PIL import Image, ImageDraw
import torch
import io
import base64
import numpy as np
from typing import TypedDict
from chest_xray.tools import bbox as bbox_tools
from chest_xray.interfaces.backend.api.gradcambox import get_gradcam_bbox
from chest_xray.interfaces.backend.api.load_model import CLASSES


class PredictionItem(TypedDict):
    label: str
    confidence: float


class BBoxItem(TypedDict):
    label: str
    bbox: tuple[float, float, float, float]


class ImageItem(TypedDict):
    label: str
    image: str


def build_bbox_images(
    image: Image.Image,
    bbox_predictions: torch.Tensor | None,
    predictions: list[PredictionItem],
) -> list[ImageItem]:
    bbox_coords: list[BBoxItem] = []
    bbox_images: list[ImageItem] = []

    if bbox_predictions is None:
        return bbox_images

    for prediction in predictions:
        if prediction["confidence"] <= 0.5:
            continue

        class_idx = CLASSES.index(prediction["label"])

        try:
            raw = bbox_predictions[class_idx].detach().cpu().tolist()
            if len(raw) != 4:
                continue

            x1_n, y1_n, x2_n, y2_n = raw
            x1_n = max(0.0, min(1.0, float(x1_n)))
            y1_n = max(0.0, min(1.0, float(y1_n)))
            x2_n = max(0.0, min(1.0, float(x2_n)))
            y2_n = max(0.0, min(1.0, float(y2_n)))

            x1 = x1_n * image.width
            y1 = y1_n * image.height
            x2 = x2_n * image.width
            y2 = y2_n * image.height

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            if w < 5 or h < 5:
                continue

            bbox_coords.append({"label": prediction["label"], "bbox": (x, y, w, h)})
        except Exception:
            continue

    for bbox_entry in bbox_coords:
        img_with_bbox = image.convert("RGB")
        draw = ImageDraw.Draw(img_with_bbox)
        bbox_tools.draw_bbox(
            draw,
            bbox_entry["label"],
            bbox_entry["bbox"][0],
            bbox_entry["bbox"][1],
            bbox_entry["bbox"][2],
            bbox_entry["bbox"][3],
        )

        buffered = io.BytesIO()
        img_with_bbox.save(buffered, format="JPEG")
        bbox_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        bbox_images.append({"label": bbox_entry["label"], "image": bbox_image_str})

    return bbox_images


def build_gradcam_images(
    image: Image.Image,
    input_tensor: torch.Tensor,
    predictions: list[PredictionItem],
    model: torch.nn.Module,
) -> list[ImageItem]:
    gbox_coords: list[BBoxItem] = []
    gbox_images: list[ImageItem] = []
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    for prediction in predictions:
        if prediction["confidence"] <= 0.5:
            continue

        class_idx = CLASSES.index(prediction["label"])
        result = get_gradcam_bbox(
            model,
            input_tensor,
            class_idx,
            target_layer=model.features[-1],
            image_np=image_np,
        )
        if result is None:
            continue

        x, y, w, h = result
        gbox_coords.append({"label": prediction["label"], "bbox": (x, y, w, h)})

    for gbox_entry in gbox_coords:
        img_with_gbox = image.convert("RGB")
        draw = ImageDraw.Draw(img_with_gbox)
        bbox_tools.draw_bbox(
            draw,
            gbox_entry["label"],
            gbox_entry["bbox"][0],
            gbox_entry["bbox"][1],
            gbox_entry["bbox"][2],
            gbox_entry["bbox"][3],
        )

        buffered = io.BytesIO()
        img_with_gbox.save(buffered, format="JPEG")
        gbox_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        gbox_images.append({"label": gbox_entry["label"], "image": gbox_image_str})

    return gbox_images
