import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import load_and_clean, aggregate_monthly
from model import build_features, train_model, forecast_future


st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📈 Sales & Demand Forecasting")
st.markdown("**Superstore Sales Dataset | Future Interns – ML Task 1**")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Settings")
months_ahead = st.sidebar.slider("Months to Forecast", 3, 24, 12)
uploaded_file = st.sidebar.file_uploader("Upload Superstore CSV", type=["csv"])

# ── Load data ─────────────────────────────────────────────
if uploaded_file is not None:
    df_raw = load_and_clean(uploaded_file)
else:
    st.info("👆 Please upload the Superstore CSV file from the sidebar to get started.")
    st.stop()

# ── Section 1: Data Preview ───────────────────────────────
st.subheader("🗂️ Dataset Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df_raw):,}")
col2.metric("Date Range", f"{df_raw['Order Date'].min().year} – {df_raw['Order Date'].max().year}")
col3.metric("Total Sales", f"${df_raw['Sales'].sum():,.0f}")

st.dataframe(df_raw[['Order Date', 'Category', 'Sub-Category', 'Region', 'Sales', 'Profit']].head(20))

st.markdown("---")

# ── Section 2: Monthly Sales Trend ───────────────────────
st.subheader("📅 Monthly Sales Trend")

monthly = aggregate_monthly(df_raw)

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(monthly['Date'], monthly['Sales'], color='steelblue', linewidth=1.8)
ax1.fill_between(monthly['Date'], monthly['Sales'], alpha=0.15, color='steelblue')
ax1.set_title("Monthly Sales Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales ($)")
ax1.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig1)

st.markdown("---")

# ── Section 3: Train Model ─────────────────────────────────
st.subheader("🤖 Model Training (Random Forest)")

monthly_feat = build_features(monthly)
model, train_preds, mae, rmse = train_model(monthly_feat)

col1, col2 = st.columns(2)
col1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
col2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")

# Actual vs Predicted chart
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(monthly_feat['Date'], monthly_feat['Sales'], label='Actual Sales', color='steelblue', linewidth=1.8)
ax2.plot(monthly_feat['Date'], train_preds, label='Predicted Sales', color='orange', linestyle='--', linewidth=1.8)
ax2.set_title("Actual vs Predicted Monthly Sales")
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales ($)")
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# ── Section 4: Future Forecast ─────────────────────────────
st.subheader(f"🔮 Sales Forecast – Next {months_ahead} Months")

future_df = forecast_future(model, monthly_feat, months_ahead=months_ahead)

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(monthly_feat['Date'], monthly_feat['Sales'], label='Historical Sales', color='steelblue', linewidth=1.8)
ax3.plot(future_df['Date'], future_df['Predicted_Sales'], label='Forecasted Sales', color='green', linewidth=2, linestyle='--', marker='o', markersize=4)
ax3.axvline(x=monthly_feat['Date'].iloc[-1], color='gray', linestyle=':', linewidth=1.2, label='Forecast Start')
ax3.set_title(f"Sales Forecast for Next {months_ahead} Months")
ax3.set_xlabel("Date")
ax3.set_ylabel("Sales ($)")
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig3)

# Forecast table
st.markdown("**Predicted Monthly Sales:**")
forecast_display = future_df[['Date', 'Predicted_Sales']].copy()
forecast_display['Date'] = forecast_display['Date'].dt.strftime('%b %Y')
forecast_display['Predicted_Sales'] = forecast_display['Predicted_Sales'].map(lambda x: f"${x:,.2f}")
forecast_display.columns = ['Month', 'Predicted Sales']
st.dataframe(forecast_display, use_container_width=True)

st.markdown("---")
st.caption("Built by a student for Future Interns – ML Task 1 | Dataset: Superstore Sales")
