import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Header
from google import genai
from google.genai import types
from pydantic import BaseModel
from supabase import create_client


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AgriMate AI Service",
    version="1.0.0",
)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL environment variable is missing"
    )

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_SERVICE_ROLE_KEY environment variable is missing"
    )


# =========================================================
# SUPABASE CLIENT
# =========================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)

gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY
    )
    if GEMINI_API_KEY
    else None
)


# =========================================================
# LOAD AI MODEL
# =========================================================

MODEL_PATH = "demand_model.pkl"

try:
    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    features = model_data["features"]

except Exception as error:
    raise RuntimeError(
        f"Failed to load AI model: {error}"
    )


# =========================================================
# REQUEST MODEL
# =========================================================

class DemandPredictionRequest(BaseModel):
    commodity_id: str


class AddressRequest(BaseModel):
    address: str


class AddressParseResult(BaseModel):
    province: str | None = None
    city: str | None = None
    district: str | None = None
    needs_confirmation: bool
    reason: str


class AddressResponse(AddressParseResult):
    address_original: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():
    return {
        "service": "AgriMate AI Service",
        "status": "running",
        "model": "Demand Forecasting Random Forest",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "supabase_connected": True,
    }


# =========================================================
# PARSE ADDRESS ENDPOINT
# =========================================================

ADDRESS_PARSE_PROMPT = """
Ekstrak alamat Indonesia berikut ke struktur administratif.

Aturan:
- province adalah nama Provinsi.
- city adalah nama Kabupaten atau Kota, termasuk awalan "Kabupaten" atau "Kota"
  bila itu bagian dari nama wilayah yang normal.
- district adalah nama Kecamatan tanpa awalan "Kecamatan".
- Gunakan nama wilayah Indonesia yang normal dan konsisten.
- Jangan menebak. Jika provinsi tidak dapat ditentukan dengan aman, isi province
  dengan null dan needs_confirmation dengan true.
- city atau district boleh null jika tidak tersedia dari alamat.
- needs_confirmation harus true bila alamat ambigu atau informasi penting tidak
  cukup untuk mengidentifikasi wilayah secara aman.
- Jangan mengarang confidence numerik.
- reason harus singkat dalam bahasa Indonesia.

Kembalikan hanya data terstruktur sesuai schema, tanpa markdown:
{
  "province": string|null,
  "city": string|null,
  "district": string|null,
  "needs_confirmation": boolean,
  "reason": string
}

Alamat:
"""


def _clean_address_value(value: str | None):
    if value is None:
        return None

    value = value.strip()
    return value or None


def parse_address_with_gemini(address: str) -> AddressParseResult:
    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is not configured: "
                "GEMINI_API_KEY environment variable is missing"
            ),
        )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=ADDRESS_PARSE_PROMPT + address,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AddressParseResult,
                temperature=0,
            ),
        )

        parsed = response.parsed
        if parsed is None:
            raise ValueError("Gemini returned no structured address data")

        if isinstance(parsed, AddressParseResult):
            result = parsed
        else:
            result = AddressParseResult(**parsed)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Address parsing with Gemini failed: {error}",
        )

    result.province = _clean_address_value(result.province)
    result.city = _clean_address_value(result.city)
    result.district = _clean_address_value(result.district)

    if result.province is None:
        result.needs_confirmation = True

    return result


@app.post("/parse-address", response_model=AddressResponse)
def parse_address(request: AddressRequest):
    address = request.address.strip()
    if not address:
        raise HTTPException(
            status_code=422,
            detail="address must not be empty",
        )

    if len(address) < 3 or len(address.split()) == 1:
        return AddressResponse(
            address_original=address,
            province=None,
            city=None,
            district=None,
            needs_confirmation=True,
            reason="Alamat belum cukup spesifik",
        )

    result = parse_address_with_gemini(address)
    return AddressResponse(
        address_original=address,
        **result.model_dump(),
    )


# =========================================================
# HELPER: GET AUTHENTICATED USER
# =========================================================

def get_authenticated_user(
    authorization: str | None
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization token required",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header",
        )

    access_token = authorization.replace(
        "Bearer ",
        "",
        1,
    ).strip()

    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Access token is empty",
        )

    try:
        user_response = supabase.auth.get_user(
            access_token
        )

        user = user_response.user

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid access token",
            )

        return user

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token",
        )


# =========================================================
# HELPER: CHECK BUYER PROFILE
# =========================================================

def check_buyer_profile(
    user_id: str
):
    try:
        response = (
            supabase
            .table("profiles")
            .select(
                "id, full_name, role"
            )
            .eq(
                "id",
                user_id,
            )
            .maybe_single()
            .execute()
        )

        profile = response.data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch profile: {error}",
        )

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Profile not found",
        )

    if profile["role"] != "BUYER":
        raise HTTPException(
            status_code=403,
            detail=(
                "Demand forecasting is only "
                "available for BUYER users"
            ),
        )

    return profile


# =========================================================
# HELPER: CHECK COMMODITY
# =========================================================

def get_commodity(
    commodity_id: str
):
    try:
        response = (
            supabase
            .table("commodities")
            .select(
                "id, name, unit, is_active"
            )
            .eq(
                "id",
                commodity_id,
            )
            .maybe_single()
            .execute()
        )

        commodity = response.data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch commodity: {error}",
        )

    if commodity is None:
        raise HTTPException(
            status_code=404,
            detail="Commodity not found",
        )

    if commodity["is_active"] is not True:
        raise HTTPException(
            status_code=400,
            detail="Commodity is not active",
        )

    return commodity


# =========================================================
# HELPER: GET BUYER HISTORY
# =========================================================

def get_demand_history(
    buyer_id: str,
    commodity_id: str,
):
    try:
        response = (
            supabase
            .table("buyer_demand_history")
            .select(
                "period_start,total_quantity"
            )
            .eq(
                "buyer_id",
                buyer_id,
            )
            .eq(
                "commodity_id",
                commodity_id,
            )
            .order(
                "period_start",
            )
            .execute()
        )

        history = response.data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch demand history: {error}",
        )

    if history is None:
        history = []

    return history


# =========================================================
# HELPER: BUILD FEATURES
# =========================================================

def build_prediction_features(
    history: list
):
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

    rolling_mean_4 = float(
        np.mean(latest_4)
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

    return (
        input_data,
        next_date,
        quantities,
    )


# =========================================================
# PREDICT DEMAND ENDPOINT
# =========================================================

@app.post("/predict-demand")
def predict_demand(
    request: DemandPredictionRequest,
    authorization: str | None = Header(
        default=None
    ),
):
    # -----------------------------------------------------
    # 1. AUTHENTICATE USER
    # -----------------------------------------------------

    user = get_authenticated_user(
        authorization
    )

    buyer_id = user.id


    # -----------------------------------------------------
    # 2. CHECK PROFILE ROLE
    # -----------------------------------------------------

    profile = check_buyer_profile(
        buyer_id
    )


    # -----------------------------------------------------
    # 3. CHECK COMMODITY
    # -----------------------------------------------------

    commodity = get_commodity(
        request.commodity_id
    )


    # -----------------------------------------------------
    # 4. GET HISTORICAL DEMAND
    # -----------------------------------------------------

    history = get_demand_history(
        buyer_id,
        request.commodity_id,
    )


    # -----------------------------------------------------
    # 5. FEATURE ENGINEERING
    # -----------------------------------------------------

    (
        input_data,
        next_date,
        quantities,
    ) = build_prediction_features(
        history
    )


    # -----------------------------------------------------
    # 6. RUN AI MODEL
    # -----------------------------------------------------

    try:
        prediction = model.predict(
            input_data
        )[0]

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )


    # -----------------------------------------------------
    # 7. CLEAN PREDICTION
    # -----------------------------------------------------

    prediction = max(
        0,
        float(prediction),
    )

    prediction = round(
        prediction,
        2,
    )


    # -----------------------------------------------------
    # 8. RESPONSE
    # -----------------------------------------------------

    return {
        "buyer": {
            "id": buyer_id,
            "name": profile["full_name"],
        },

        "commodity": {
            "id": commodity["id"],
            "name": commodity["name"],
            "unit": commodity["unit"],
        },

        "prediction": {
            "period_start":
                str(next_date.date()),

            "predicted_quantity":
                prediction,

            "unit":
                commodity["unit"],
        },

        "history": {
            "count":
                len(history),

            "latest_quantity":
                quantities[-1],

            "latest_period":
                history[-1]["period_start"],
        },
    }