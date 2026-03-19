from lightgbm import LGBMRegressor

def forecast(df):

    features = [
        "product_id","time_index","qtr",
        "lag1","lag2","lag4",
        "rolling_mean4",
        "growth","trend",
        "pipeline_ratio",
        "SCMS_units","VMS_units",
        "bigdeal_flag"
    ]

    max_t = df["time_index"].max()

    train = df[df["time_index"] < max_t]
    test = df[df["time_index"] == max_t]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64
    )

    model.fit(train[features], train["Bookings"])

    ml = model.predict(test[features])
    pipe = test["VMS_units"] * 0.7

    final = 0.8 * ml + 0.2 * pipe

    test["Forecast"] = final

    return test, model, features