# ChestXrays

ChestXrays is an application that detects and classifies possible diseases from chest X-ray images using a trained machine learning model.

The project includes a frontend for uploading images, a backend API for processing predictions, and a REST API that connects both parts of the application.

---

## Overview

This project contains:

* A frontend where users can upload chest X-ray images
* A backend API that loads a trained model and returns predictions
* A REST API for communication between the frontend and backend
* Docker Compose configuration to run the full application easily

Users upload a chest X-ray image through the frontend. The image is sent to the backend API, where the trained model analyzes it and returns a prediction result.

---

## Requirements

Make sure the following tools are installed on your system:

* Python
* Pipenv
* Node.js
* npm
* Docker Engine or Docker Desktop
* Docker Compose

---

## Installation

Clone the repository:

```
git clone https://github.com/Chest-Xray/AML.git
cd AML
```

Install the Python dependencies:

```
pipenv install -r requirements.txt
```

---


Before starting the application, make sure the trained model file is available locally at the path expected by the backend:

```
chest_xray/interfaces/backend/api/densenet_pretrained_epoch10.pth
```

If this file is missing, the backend may fail to start or may not be able to make predictions.

---

## Running the Application

Navigate to the root folder of the project, where the `docker-compose.yml` file is located.

Start the application with:

```
docker compose up --build
```

Docker Compose will build and start the required containers.

After the containers have started, the application will be available at:

```
Frontend: http://localhost:3000
Backend:  http://localhost:8000
API docs: http://localhost:8000/docs
```

---

## Using the Application

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

To make a prediction:

1. Click **Choose File**
2. Select a chest X-ray image from your computer
3. Click the **Upload** button
4. Wait for the prediction result to appear

The backend will process the image and return the predicted disease label with a confidence score.

---

## API Endpoints

The backend API documentation is available at:

```
http://localhost:8000/docs
```

## Project Structure

A simplified overview of the project structure:

```
AML/
├── chest_xray/
│   └── interfaces/
│       ├── backend/
│       │   └── api/
│       │       └── densenet_pretrained_epoch10.pth
│       └── frontend/
├── docker-compose.yml
├── requirements.txt
└── README.md
```