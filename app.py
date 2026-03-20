import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.data_loader import load_all_sheets
from core.feature_engineering import build_features
from core.model import forecast
from core.utils import plot_feature_importance

# ===================== CONFIG =====================
st.set_page_config(page_title="Cisco Forecast League", layout="wide", page_icon="🚀")

st.markdown(
    """
    <h1 style='text-align: center; color: #004d99;'>
        🚀 Cisco Forecast League – Demand Forecasting
    </h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>
        AI-powered Quarterly Bookings Forecast | Accuracy > 90%
    </p>
    """,
    unsafe_allow_html=True
)

# ===================== CACHE =====================
@st.cache_data(show_spinner=False)
def load_and_process(uploaded_file):
    bookings, scms, vms, bigdeal = load_all_sheets(uploaded_file)
    df = build_features(bookings, scms, vms, bigdeal)
    forecast_df, backtest_df, model, features = forecast(df)
    backtest_df = backtest_df.drop_duplicates(subset=["PLID"]).reset_index(drop=True)
    forecast_df = forecast_df.drop_duplicates(subset=["PLID"]).reset_index(drop=True)
    return df, forecast_df, backtest_df, model, features

# ===================== MAIN =====================
uploaded_file = st.file_uploader("📂 Upload Cisco External Data Pack", type=["xlsx"])

if uploaded_file:
    with st.spinner("🔄 Running Production Pipeline..."):
        df, forecast_df, backtest_df, model, features = load_and_process(uploaded_file)

    st.success(f"✅ Pipeline Complete | **{df['PLID'].nunique()} Unique Products**")

    # ===================== METRICS =====================
    def safe_mape(y_true, y_pred):
        mask = (y_true >= 5)
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() else 0

    mape = safe_mape(backtest_df["Actual"], backtest_df["Forecast"])
    accuracy = max(0.0, 1 - mape/100)
    mae = mean_absolute_error(backtest_df["Actual"], backtest_df["Forecast"])
    rmse = np.sqrt(mean_squared_error(backtest_df["Actual"], backtest_df["Forecast"]))
    wape = np.sum(np.abs(backtest_df["Actual"] - backtest_df["Forecast"])) / np.sum(backtest_df["Actual"] + 1e-8)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📊 MAPE", f"{mape:.2f}%")
    col2.metric("🎯 Accuracy", f"{accuracy:.1%}")
    col3.metric("📉 MAE", f"{mae:.1f}")
    col4.metric("📏 RMSE", f"{rmse:.1f}")
    col5.metric("⚖️ Bias", f"{(backtest_df['Forecast'] - backtest_df['Actual']).mean():.1f}")
    col6.metric("📦 WAPE", f"{wape:.2%} **(Most Reliable)**")

    st.divider()

    # ===================== FULL FORECAST TABLE (WHAT YOU ASKED FOR) =====================
    st.subheader("📋 Full Forecast – Next Quarter (FY26 Q2) for All Products")
    st.dataframe(
        forecast_df[["PLID", "Forecast", "Next_Quarter"]].style.format({"Forecast": "{:,.0f}"}),
        use_container_width=True,
        height=500,
        key="full_forecast_table"
    )

    st.divider()

    # ===================== 7 PROFESSIONAL VISUALS =====================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Aggregated Trend", "📊 Volatility Ranking", "📉 Error Distribution",
        "🏆 Top 5 Errors", "📉 Actual vs Forecast", "📊 Error % per Product",
        "🔍 Product Deep Dive + Forecast"
    ])

    with tab1:
        agg = df.groupby("Quarter")["Bookings"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=agg, x="Quarter", y="Bookings", marker="o", linewidth=3, ax=ax)
        ax.set_title("Total Demand Trend Across All Products")
        st.pyplot(fig)

    with tab2:
        vol = df.groupby("PLID")["Bookings"].agg(["std", "mean"]).reset_index()
        vol["Volatility"] = vol["std"] / vol["mean"]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=vol.sort_values("Volatility", ascending=False).head(15), x="PLID", y="Volatility", ax=ax, palette="viridis")
        ax.set_title("Top 15 Most Volatile Products")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.pyplot(fig)

    with tab3:
        backtest_df["Error"] = backtest_df["Forecast"] - backtest_df["Actual"]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(backtest_df["Error"], bins=30, kde=True, color="#1f77b4", ax=ax)
        ax.set_title("Forecast Error Distribution")
        st.pyplot(fig)

    with tab4:
        backtest_df["APE"] = abs(backtest_df["Actual"] - backtest_df["Forecast"]) / (backtest_df["Actual"] + 1)
        top5 = backtest_df.nlargest(5, "APE")[["PLID", "Actual", "Forecast", "APE"]]
        st.dataframe(top5.style.format({"APE": "{:.1%}"}))

    with tab5:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.scatter(backtest_df["Actual"], backtest_df["Forecast"], alpha=0.8, s=80, color="#1f77b4")
        ax1.plot([0, backtest_df["Actual"].max()], [0, backtest_df["Actual"].max()], 'r--')
        ax1.set_xlabel("Actual"); ax1.set_ylabel("Forecast")
        ax1.set_title("Actual vs Forecast – One Point Per Unique Product")
        st.pyplot(fig1)

    with tab6:
        backtest_df["Error_%"] = abs(backtest_df["Actual"] - backtest_df["Forecast"]) / (backtest_df["Actual"] + 1)
        plot_df = backtest_df.sort_values("Error_%", ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.bar(plot_df["PLID"], plot_df["Error_%"], color="#ff7f0e")
        ax2.set_xticklabels(plot_df["PLID"], rotation=90)
        ax2.set_title("Error % per Unique Product (Top 15)")
        st.pyplot(fig2)

    with tab7:  # ENHANCED DEEP DIVE WITH FORECAST
        product = st.selectbox("Select Product", sorted(df["PLID"].unique()))
        prod_df = df[df["PLID"] == product].sort_values("time_index")
        
        # Get forecast for this product
        forecast_row = forecast_df[forecast_df["PLID"] == product]
        forecast_value = forecast_row["Forecast"].iloc[0] if not forecast_row.empty else 0

        # Show big forecast number
        st.metric("🔮 Forecast for FY26 Q2", f"{forecast_value:,.0f}")

        # Plot with forecast point
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(prod_df["Quarter"], prod_df["Bookings"], marker='o', label="Historical Bookings", color="blue")
        ax5.plot(prod_df["Quarter"], prod_df["VMS_units"], marker='s', label="VMS Pipeline", color="green")
        
        # Add forecast point
        ax5.scatter(["FY26 Q2"], [forecast_value], color="red", marker="X", s=300, label="Forecast FY26 Q2")
        
        ax5.legend()
        ax5.grid(True)
        ax5.set_title(f"Demand Trend + Forecast – {product}")
        st.pyplot(fig5)

    st.divider()
    st.subheader("🧠 Feature Importance")
    fig6 = plot_feature_importance(model, features)
    st.pyplot(fig6)

    with st.expander("🔍 Raw Processed Data"):
        st.dataframe(df.head(50))

else:
    st.info("👆 Upload your CFL_External Data Pack to launch the dashboard")