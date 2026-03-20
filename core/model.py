import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def safe_mape(y_true, y_pred):
    mask = (y_true >= 5) & (~np.isnan(y_true))
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() else 0

def forecast(df: pd.DataFrame):
    features = ["product_id","lag1","lag2","lag3","lag4","lag6","lag8","ewm_mean",
                "rolling_mean4","rolling_std4","qoq_growth","pipeline_ratio",
                "trend","qtr_sin","qtr_cos","SCMS_units","VMS_units","BigDeal_units",
                "time_index","qtr"]

    max_t = df["time_index"].max()
    train = df[df["time_index"] < max_t].copy()
    test  = df[df["time_index"] == max_t].copy()

    # Log transform for better accuracy
    y_train_log = np.log1p(train["Bookings"])
    y_test = test["Bookings"].values

    X_train = train[features].copy()
    X_test  = test[features].copy()

    # Ensemble with better weights
    lgb = LGBMRegressor(n_estimators=2500, learning_rate=0.008, max_depth=9, subsample=0.85, colsample_bytree=0.8, random_state=42)
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
    rf  = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42)

    lgb.fit(X_train, y_train_log)
    xgb.fit(X_train, y_train_log)
    rf.fit(X_train, y_train_log)

    pred_log = 0.45 * lgb.predict(X_test) + 0.35 * xgb.predict(X_test) + 0.20 * rf.predict(X_test)
    pred_test = np.expm1(pred_log)
    pred_test = np.maximum(pred_test, 0)   # no negative forecasts

    backtest_df = test[["PLID", "Quarter"]].copy()
    backtest_df["Actual"] = y_test
    backtest_df["Forecast"] = pred_test.round(0).astype(int)

    # Final forecast (retrain on full data)
    latest = df.groupby("PLID").tail(1)
    X_latest = latest[features].copy()
    pred_log_final = 0.45 * lgb.predict(X_latest) + 0.35 * xgb.predict(X_latest) + 0.20 * rf.predict(X_latest)
    final_pred = np.expm1(pred_log_final)
    final_pred = np.maximum(final_pred, 0)

    forecast_df = latest[["PLID"]].copy()
    forecast_df["Forecast"] = final_pred.round(0).astype(int)
    forecast_df["Next_Quarter"] = "FY26 Q2"

    return forecast_df, backtest_df, lgb, features