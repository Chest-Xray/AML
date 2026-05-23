from fastapi import FastAPI

app = FastAPI()


@app.get("/prediction")
def read_root():
    return {"Hello": "World"}
