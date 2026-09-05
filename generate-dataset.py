import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

BUYER_ID = "demo-buyer-001"
COMMODITY_ID = "24fdb2bc-59a7-4c98-ade8-e831c494c9c4"
# Cabai Rawit Merah

start_date = datetime(2025, 1, 6)

rows = []

base_demand = 300

for week in range(80):
    date = start_date + timedelta(weeks=week)

    # trend permintaan meningkat sedikit
    trend = week * 1.5

    # pola musiman sederhana
    seasonal = 30 * np.sin(2 * np.pi * week / 12)

    # random fluctuation
    noise = np.random.normal(0, 15)

    quantity = base_demand + trend + seasonal + noise

    quantity = max(50, round(quantity, 2))

    rows.append({
        "buyer_id": BUYER_ID,
        "commodity_id": COMMODITY_ID,
        "period_start": date.strftime("%Y-%m-%d"),
        "total_quantity": quantity
    })

df = pd.DataFrame(rows)

df.to_csv(
    "demand_history_demo.csv",
    index=False
)

print(df.head())
print()
print(f"Dataset created: {len(df)} rows")