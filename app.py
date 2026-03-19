import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from core.data_loader import load_all_sheets
from core.feature_engineering import build_features
from core.model import forecast
from core.utils import plot_feature_importance

# ---------------- UI CONFIG ---------------- #
st.set_page_config(
    page_title="Cisco Forecast League Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- HEADER ---------------- #
st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>Cisco Forecast League Dashboard</h1>
    <p style='text-align: center;'>AI-driven Demand Forecasting for 30 Products</p>
    """,
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader("📂 Upload Cisco Data Pack", type=["xlsx"])

if file:

    # ---------------- LOAD DATA ---------------- #
    bookings, scms, vms, bigdeal = load_all_sheets(file)

    st.success("✅ Data Loaded Successfully")

    # ---------------- FEATURE ENGINEERING ---------------- #
    df = build_features(bookings, scms, vms, bigdeal)

    # ---------------- FORECAST ---------------- #
    result, model, features = forecast(df)

    # ---------------- VALIDATION SET ---------------- #
    max_time = df["time_index"].max()
    test = df[df["time_index"] == max_time]

    compare = result.copy()
    compare["Actual"] = test["Bookings"].values

    # ---------------- METRICS ---------------- #
    mape = mean_absolute_percentage_error(compare["Actual"], compare["Forecast"])
    mae = mean_absolute_error(compare["Actual"], compare["Forecast"])

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 MAPE", f"{mape:.2%}")
    col2.metric("📉 MAE", f"{mae:.2f}")
    col3.metric("📦 Products Forecasted", len(compare))

    st.divider()

    # ---------------- FORECAST TABLE ---------------- #
    st.subheader("📈 Final Forecast")

    st.dataframe(compare[["PLID", "Actual", "Forecast"]])

    st.download_button(
        "⬇️ Download Forecast",
        compare.to_csv(index=False),
        "forecast.csv"
    )

    st.divider()

    # ---------------- FORECAST VS ACTUAL ---------------- #
    st.subheader("📉 Forecast vs Actual")

    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(compare["Actual"].values, label="Actual", marker='o')
    ax1.plot(compare["Forecast"].values, label="Forecast", marker='o')
    ax1.legend()
    ax1.set_title("Forecast vs Actual")

    st.pyplot(fig1)

    # ---------------- ERROR ANALYSIS ---------------- #
    st.subheader("📊 Error per Product")

    compare["Error_%"] = abs(compare["Actual"] - compare["Forecast"]) / (compare["Actual"] + 1)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.bar(compare["PLID"], compare["Error_%"])
    ax2.set_xticklabels(compare["PLID"], rotation=90)
    ax2.set_title("Error % per Product")

    st.pyplot(fig2)

    st.divider()

    # ---------------- VMS VS BOOKINGS ---------------- #
    st.subheader("🔗 Pipeline (VMS) vs Demand")

    fig3, ax3 = plt.subplots()
    ax3.scatter(df["VMS_units"], df["Bookings"])
    ax3.set_xlabel("VMS Units")
    ax3.set_ylabel("Bookings")

    st.pyplot(fig3)

    st.divider()

    # ---------------- VOLATILITY ---------------- #
    st.subheader("📈 Demand Volatility")

    volatility = df.groupby("PLID")["Bookings"].std() / df.groupby("PLID")["Bookings"].mean()

    fig4, ax4 = plt.subplots(figsize=(10,5))
    volatility.plot(kind="bar", ax=ax4)
    ax4.set_title("Product Volatility")

    st.pyplot(fig4)

    st.divider()

    # ---------------- PRODUCT DEEP DIVE ---------------- #
    st.subheader("📦 Product Deep Dive")

    product = st.selectbox("Select Product", df["PLID"].unique())

    prod_df = df[df["PLID"] == product]

    fig5, ax5 = plt.subplots(figsize=(10,5))
    ax5.plot(prod_df["time_index"], prod_df["Bookings"], label="Bookings")
    ax5.plot(prod_df["time_index"], prod_df["VMS_units"], label="VMS")
    ax5.legend()
    ax5.set_title(f"Demand Trend - {product}")

    st.pyplot(fig5)

    st.divider()

    # ---------------- FEATURE IMPORTANCE ---------------- #
    st.subheader("🧠 Model Feature Importance")

    fig6 = plot_feature_importance(model, features)
    st.pyplot(fig6)

    st.divider()

    # ---------------- RAW DATA PREVIEW ---------------- #
    with st.expander("🔍 View Processed Data"):
        st.dataframe(df.head(50))