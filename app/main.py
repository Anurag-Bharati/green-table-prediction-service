
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from app.model.model import predict_orders
from app.model.model import __version__ as model_version
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
core_updating = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataIn(BaseModel):
    data: list


class PredictionOut(BaseModel):
    predicted_orders: List[float]
    success: bool


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut, status_code=status.HTTP_200_OK)
def predict(payload: DataIn, response: Response):
    orders = predict_orders(payload.data)
    if orders is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"success": False, "predicted_orders": []}
    return {"success": True, "predicted_orders": orders.tolist()}


