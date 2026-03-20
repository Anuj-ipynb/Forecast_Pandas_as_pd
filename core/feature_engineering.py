import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_features(bookings, scms, vms, bigdeal):
    df = bookings.merge(scms, on=["PLID", "Quarter"], how="left")
    df = df.merge(vms, on=["PLID", "Quarter"], how="left")
    df = df.merge(bigdeal, on=["PLID", "Quarter"], how="left")
    df.fillna(0, inplace=True)

    # Time features + seasonality
    df["year"] = df["Quarter"].str.extract(r"FY(\d{2})").astype(int) + 2000
    df["qtr"] = df["Quarter"].str.extract(r"Q(\d)").astype(int)
    df["time_index"] = df["year"] * 4 + df["qtr"]
    df["qtr_sin"] = np.sin(2 * np.pi * df["qtr"] / 4)
    df["qtr_cos"] = np.cos(2 * np.pi * df["qtr"] / 4)

    df = df.sort_values(["PLID", "time_index"])

    def add_features(group):
        g = group.copy()
        for lag in [1,2,3,4,6,8]:
            g[f"lag{lag}"] = g["Bookings"].shift(lag)
        g["ewm_mean"] = g["Bookings"].ewm(span=4).mean()
        g["rolling_mean4"] = g["Bookings"].rolling(4, min_periods=1).mean()
        g["rolling_std4"] = g["Bookings"].rolling(4, min_periods=1).std().fillna(0)
        g["qoq_growth"] = (g["lag1"] - g["lag2"]) / (g["lag2"] + 1)
        g["pipeline_ratio"] = g["VMS_units"] / (g["rolling_mean4"] + 1)
        g["trend"] = g["Bookings"].rolling(4).mean().diff()
        return g

    df = df.groupby("PLID", group_keys=False).apply(add_features)
    df.fillna(0, inplace=True)

    encoder = LabelEncoder()
    df["product_id"] = encoder.fit_transform(df["PLID"])
    return df