from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
from chest_xray.interfaces.backend.api.load_model import get_model, CLASSES
from chest_xray.interfaces.backend.api.helpers import (
    build_bbox_images,
    build_gradcam_images,
)
from typing import TypedDict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, transform, device = get_model()

class PredictionItem(TypedDict):
    label: str
    confidence: float


class BoxItem(TypedDict):
    label: str
    box: tuple[float, float, float, float]


class ImageItem(TypedDict):
    label: str
    image: str


@app.post("/prediction")
async def prediction(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    image_tensor = transform(image)
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("transform(image) must return a torch.Tensor")
    input_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    bbox_predictions = None
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        class_logits, bbox_predictions = output[0], output[1]
    else:
        class_logits = output

    if not isinstance(class_logits, torch.Tensor):
        class_logits = torch.tensor(class_logits)
    if class_logits.dim() == 2:
        class_logits = class_logits[0]

    probs = torch.sigmoid(class_logits).cpu()

    predictions: list[PredictionItem] = []
    for i, p in enumerate(probs.tolist()):
        predictions.append({"label": CLASSES[i], "confidence": float(p)})
        
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    bbox_predictions = bbox_predictions[0]

    bbox_images = build_bbox_images(image, bbox_predictions, predictions)
    gbox_images = build_gradcam_images(image, input_tensor, predictions, model)

    return {
        "predictions": predictions,
        "bbox_images": bbox_images,
        "gradcam_images": gbox_images,
    }
