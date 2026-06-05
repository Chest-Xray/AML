from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import models, transforms
import io

from load_model import get_model, CLASSES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model, transform, device = get_model()


@app.post("/prediction")
async def create_prediction(file: UploadFile = File(...)):
    print("Received file:", file.filename)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs)[0]

    all_predictions = []

    for class_name, probability in zip(CLASSES, probabilities):
        all_predictions.append(
            {"label": class_name, "confidence": float(probability.item())}
        )

    predictions = sorted(
        all_predictions, key=lambda item: item["confidence"], reverse=True
    )

    return {
        "filename": file.filename,
        "predictions": predictions,
    }
