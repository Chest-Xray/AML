# ChestXrays

This project detects and classifies diseases from chest X-ray images and returns a prediction.

---

## Overview

This project contains:

- A frontend for uploading chest X-ray images
- A backend API that loads the trained model and makes predictions
- A REST API for communication between the frontend and backend

Users can upload an image through the frontend. The image is sent to the backend, where the model generates predictions and sends the result back to the frontend.

---

## Requirements

Make sure you have the following installed:

- Python
- Pipenv
- Node.js
- npm

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Chest-Xray/AML.git
cd <your_project_folder>
```

Install the Python dependencies:

```bash
pipenv install -r requirements.txt
```

## Running the backend

Navigate to the root folder

Start the backend server:

```bash
python -m uvicorn chest_xray.interfaces.backend.api.api:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

## Running the frontend

Open a second terminal and navigate to the frontend folder:

```bash
cd chest_xray/interfaces/frontend/image-uploader
```

Install required npm packages for the backend:
```bash
npm install
```

Start the frontend development server:

```bash
npm run dev
```

The frontend will run at:

```text
http://localhost:3000
```

Open this URL in your browser to upload chest X-ray images and receive predictions.

---

## Usage

1. Start the backend server
2. Start the frontend server
3. Open the frontend in your browser via the link in the terminal
4. Click the `browse` button to select an image
5. Click the `upload` button to upload the image to the backend
6. Use your browser's inspect tool to view the console
7. A json string will be returned with the predictions

---

## API Endpoint

### `POST /prediction`

Uploads a chest X-ray image and returns model predictions, bbox images and gradcam images.

Example response:

```json
{
  "bbox_images": [
    {
      "label": "Pneumonia",
      "image": "base64 string"
    }
  ],
  "gradcam_images": [
    {
      "label": "Pneumonia",
      "image": "base64 string"
    }
  ],
  "predictions": [
    {
      "label": "Pneumonia",
      "confidence": 0.87
    }
  ]
}
```

---

## Model file

The trained model file may be too large to store directly in GitHub.

Make sure the model file is available locally at the path expected by the backend before starting the API.

Expected model path:

```text
chest_xray/data/models/densenet_pretrained_epoch10.pth
```

If the model file is not present, the backend will not be able to start or make predictions.

---

## Notes

- The backend must be running before using the frontend.
- The frontend sends uploaded images to the backend API.
- The backend returns the model predictions as JSON.
