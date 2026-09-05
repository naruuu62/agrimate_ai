import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


DATASET_PATH = "demand_history_demo.csv"
MODEL_PATH = "demand_model.pkl"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATASET_PATH)

df["period_start"] = pd.to_datetime(
    df["period_start"]
)

df = df.sort_values(
    "period_start"
).reset_index(drop=True)


# =========================
# FEATURE ENGINEERING
# =========================

df["lag_1"] = df["total_quantity"].shift(1)
df["lag_2"] = df["total_quantity"].shift(2)
df["lag_3"] = df["total_quantity"].shift(3)
df["lag_4"] = df["total_quantity"].shift(4)

df["rolling_mean_4"] = (
    df["total_quantity"]
    .shift(1)
    .rolling(window=4)
    .mean()
)

df["week_of_year"] = (
    df["period_start"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["month"] = df["period_start"].dt.month


# hapus row yang belum punya lag
df = df.dropna().reset_index(drop=True)


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "rolling_mean_4",
    "week_of_year",
    "month",
]

TARGET = "total_quantity"


X = df[FEATURES]
y = df[TARGET]


# =========================
# TIME-BASED SPLIT
# =========================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Train size:", len(X_train))
print("Test size :", len(X_test))


# =========================
# TRAIN MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# =========================
# EVALUATION
# =========================

prediction = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    prediction,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction,
    )
)

print()
print("MODEL EVALUATION")
print("----------------")
print(f"MAE  : {mae:.2f} kg")
print(f"RMSE : {rmse:.2f} kg")


# =========================
# SAVE MODEL
# =========================

model_data = {
    "model": model,
    "features": FEATURES,
    "mae": mae,
    "rmse": rmse,
}

joblib.dump(
    model_data,
    MODEL_PATH,
)

print()
print(
    f"Model saved as: {MODEL_PATH}"
)


# =========================
# PREDICT NEXT WEEK
# =========================

latest = df.iloc[-1]

next_date = (
    latest["period_start"]
    + pd.Timedelta(weeks=1)
)

next_features = pd.DataFrame([
    {
        "lag_1": latest["total_quantity"],
        "lag_2": latest["lag_1"],
        "lag_3": latest["lag_2"],
        "lag_4": latest["lag_3"],

        "rolling_mean_4": np.mean([
            latest["total_quantity"],
            latest["lag_1"],
            latest["lag_2"],
            latest["lag_3"],
        ]),

        "week_of_year":
            int(next_date.isocalendar().week),

        "month":
            next_date.month,
    }
])


next_prediction = model.predict(
    next_features
)[0]

print()
print("NEXT WEEK FORECAST")
print("------------------")
print(
    "Period:",
    next_date.date()
)

print(
    f"Predicted demand: "
    f"{next_prediction:.2f} kg"
)