import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="AgriMate AI Service",
    version="1.0.0",
)


# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "demand_model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


# =========================
# REQUEST MODEL
# =========================

class DemandPredictionRequest(BaseModel):
    history: list[float]


# =========================
# RESPONSE MODEL
# =========================

class DemandPredictionResponse(BaseModel):
    predicted_quantity: float
    unit: str
    history_count: int


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def root():
    return {
        "service": "AgriMate AI Service",
        "status": "running",
    }


# =========================
# PREDICT DEMAND
# =========================

@app.post(
    "/predict-demand",
    response_model=DemandPredictionResponse,
)
def predict_demand(
    request: DemandPredictionRequest,
):
    history = request.history

    if len(history) < 4:
        raise HTTPException(
            status_code=400,
            detail="Minimum 4 historical periods are required",
        )

    # 4 data terakhir
    latest_4 = history[-4:]

    lag_1 = latest_4[-1]
    lag_2 = latest_4[-2]
    lag_3 = latest_4[-3]
    lag_4 = latest_4[-4]

    rolling_mean_4 = np.mean(latest_4)

    # Untuk MVP:
    # bulan dan week_of_year sementara menggunakan tanggal sekarang.
    now = pd.Timestamp.now()

    input_data = pd.DataFrame(
        [
            {
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_4": lag_4,
                "rolling_mean_4": rolling_mean_4,
                "week_of_year": int(
                    now.isocalendar().week
                ),
                "month": now.month,
            }
        ]
    )

    input_data = input_data[features]

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        float(prediction),
    )

    return DemandPredictionResponse(
        predicted_quantity=round(
            prediction,
            2,
        ),
        unit="kg",
        history_count=len(history),
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )