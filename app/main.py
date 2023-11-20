from fastapi import FastAPI, UploadFile
from predict import predict


app = FastAPI()


@app.post("/atomxyz/")
async def pred_xyz(file: UploadFile, model: str = "graph_transformer"):
    data = [_.decode() for _ in file.file.readlines()]
    return predict(data, model)
