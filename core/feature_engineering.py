import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_features(bookings, scms, vms, bigdeal):

    # BOOKINGS
    q_cols = [c for c in bookings.columns if isinstance(c, str) and "FY" in c]

    b = bookings.melt(
        id_vars=[bookings.columns[1], bookings.columns[2]],
        value_vars=q_cols,
        var_name="Quarter",
        value_name="Bookings"
    )

    b = b.rename(columns={
        bookings.columns[1]: "PLID",
        bookings.columns[2]: "Category"
    })

    b["Bookings"] = pd.to_numeric(b["Bookings"], errors="coerce")
    b = b.dropna(subset=["PLID", "Bookings"])

    # SCMS
    s = scms.melt(
        id_vars=[scms.columns[1], scms.columns[2]],
        var_name="Quarter",
        value_name="SCMS_units"
    )

    s = s.rename(columns={
        scms.columns[1]: "PLID",
        scms.columns[2]: "Segment"
    })

    s = s.groupby(["PLID","Quarter"])["SCMS_units"].sum().reset_index()

    # VMS
    v = vms.melt(
        id_vars=[vms.columns[1], vms.columns[2]],
        var_name="Quarter",
        value_name="VMS_units"
    )

    v = v.rename(columns={
        vms.columns[1]: "PLID",
        vms.columns[2]: "VMS_Name"
    })

    v["VMS_units"] = pd.to_numeric(v["VMS_units"], errors="coerce")
    v = v.groupby(["PLID","Quarter"])["VMS_units"].sum().reset_index()

    # BIG DEAL
    bd = bigdeal.copy()
    cols = bd.columns[3:10]
    quarters = bd.loc[0, cols].tolist()

    bd_clean = bd.loc[1:, ["PLID Masked"] + list(cols)]
    bd_clean.columns = ["PLID"] + quarters

    bd_long = bd_clean.melt(
        id_vars=["PLID"],
        var_name="Quarter",
        value_name="BigDeal_units"
    )

    bd_long["BigDeal_units"] = pd.to_numeric(bd_long["BigDeal_units"], errors="coerce")

    # MERGE
    df = b.merge(s, on=["PLID","Quarter"], how="left")
    df = df.merge(v, on=["PLID","Quarter"], how="left")
    df = df.merge(bd_long, on=["PLID","Quarter"], how="left")

    df.fillna(0, inplace=True)

    # TIME
    df["Quarter"] = df["Quarter"].str.replace("FY","20").str.replace(" ","")
    df["year"] = df["Quarter"].str[:4].astype(int)
    df["qtr"] = df["Quarter"].str[-1].astype(int)
    df["time_index"] = df["year"]*4 + df["qtr"]

    # FEATURES
    df["lag1"] = df.groupby("PLID")["Bookings"].shift(1)
    df["lag2"] = df.groupby("PLID")["Bookings"].shift(2)
    df["lag4"] = df.groupby("PLID")["Bookings"].shift(4)

    df["rolling_mean4"] = df.groupby("PLID")["Bookings"].transform(lambda x: x.rolling(4).mean())

    df["growth"] = (df["lag1"] - df["lag2"]) / (df["lag2"] + 1)
    df["trend"] = df["lag1"] - df["lag4"]

    df["pipeline_ratio"] = df["VMS_units"] / (df["rolling_mean4"] + 1)
    df["bigdeal_flag"] = (df["BigDeal_units"] > 0).astype(int)

    enc = LabelEncoder()
    df["product_id"] = enc.fit_transform(df["PLID"])

    df = df.dropna()

    return df