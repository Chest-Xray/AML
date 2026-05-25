from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.post("/prediction")
async def create_prediction(file: UploadFile):
    # send the file to the model and get the prediction
    prediction = "some prediction" 
    return {"filename": file.filename, "prediction": prediction}

