from fastapi import FastAPI, UploadFile
from predict import predict


app = FastAPI()


@app.post("/atomxyz/")
async def pred_xyz(file: UploadFile):
    data = [_.decode() for _ in file.file.readlines()]
    return predict(data)
