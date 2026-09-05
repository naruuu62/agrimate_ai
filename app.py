import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client


app = FastAPI(
    title="AgriMate AI Service",
    version="1.0.0",
)


# =========================
# SUPABASE
# =========================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ[
    "SUPABASE_SERVICE_ROLE_KEY"
]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)


# =========================
# LOAD MODEL
# =========================

model_data = joblib.load(
    "demand_model.pkl"
)

model = model_data["model"]
features = model_data["features"]


# =========================
# REQUEST
# =========================

class DemandPredictionRequest(BaseModel):
    buyer_id: str
    commodity_id: str


# =========================
# ENDPOINT
# =========================

@app.post("/predict-demand")
def predict_demand(
    request: DemandPredictionRequest,
):
    response = (
        supabase
        .table("buyer_demand_history")
        .select(
            "period_start,total_quantity"
        )
        .eq(
            "buyer_id",
            request.buyer_id,
        )
        .eq(
            "commodity_id",
            request.commodity_id,
        )
        .order(
            "period_start",
        )
        .execute()
    )

    history = response.data

    if len(history) < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                "Minimum 4 historical periods "
                "are required"
            ),
        )

    quantities = [
        float(row["total_quantity"])
        for row in history
    ]

    latest_4 = quantities[-4:]

    lag_1 = latest_4[-1]
    lag_2 = latest_4[-2]
    lag_3 = latest_4[-3]
    lag_4 = latest_4[-4]

    rolling_mean_4 = np.mean(
        latest_4
    )

    last_period = pd.Timestamp(
        history[-1]["period_start"]
    )

    next_date = (
        last_period
        + pd.Timedelta(weeks=1)
    )

    input_data = pd.DataFrame(
        [
            {
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_4": lag_4,
                "rolling_mean_4":
                    rolling_mean_4,
                "week_of_year":
                    int(
                        next_date
                        .isocalendar()
                        .week
                    ),
                "month":
                    next_date.month,
            }
        ]
    )

    input_data = input_data[
        features
    ]

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        float(prediction),
    )

    return {
        "buyer_id":
            request.buyer_id,

        "commodity_id":
            request.commodity_id,

        "prediction_period":
            str(next_date.date()),

        "predicted_quantity":
            round(prediction, 2),

        "unit":
            "kg",

        "history_count":
            len(history),
    }