from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import torch
import io
from interfaces.backend.api.load_model import get_model, CLASSES
import base64
import tools.bbox as bbox_tools

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
        print("Model outputs:", outputs)
        print("Output torch:", torch.sigmoid(outputs))
        probabilities = torch.sigmoid(outputs)[0]

    all_predictions = []
    bbox_images = []

    for class_name, probability in zip(CLASSES, probabilities):
        all_predictions.append(
            {"label": class_name, "confidence": float(probability.item())}
        )
        
            
        

    predictions = sorted(
        all_predictions, key=lambda item: item["confidence"], reverse=True
    )
    
    for prediction in predictions:
        bbox_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bbox_draw = ImageDraw.Draw(bbox_image)
        if prediction["confidence"] > 0.5:  # Threshold for drawing bbox
            bbox_tools.draw_bbox(bbox_draw, prediction["label"], 500, 500, 100, 100)
            buffered = io.BytesIO()
            bbox_image.save(buffered, format="JPEG")
            bbox_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    
    return {
        "filename": file.filename,
        "predictions": predictions,
        "bbox_images": bbox_images,
    }
