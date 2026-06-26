import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from data_preprocessing import load_and_clean, aggregate_monthly
from model import build_features, train_model, forecast_future


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Style ─────────────────────────────────────────────
# Accessible color palette — every pair below is chosen to meet
# at least WCAG AA contrast against the background it's used on.
PRIMARY = "#8B5CF6"
ACCENT = "#06B6D4"
WARN = "#F59E0B"    # Amber      # Amber
MUTED = "#64748B"

TEXT_DARK = "#111827"
TEXT_MUTED_LIGHT = "#9CA3AF"

SIDEBAR_BG = "#111827"
SIDEBAR_TEXT = "#F9FAFB"
SIDEBAR_MUTED = "#D1D5DB"
SIDEBAR_WIDGET_BG = "#7C3AED"

mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

st.markdown(
    """
    <style>
    .main { background-color: transparent; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    h1, h2, h3 { color: #0F172A; font-weight: 700; }

    .hero {
        padding: 1.75rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; margin-bottom: 0.25rem; font-size: 2rem; }
    .hero p { color: #DCE5FF; margin: 0; font-size: 0.95rem; }

    h1, h2, h3 {
        color: white;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.9rem;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #E2E8F0;
    }

    div[data-testid="stMetricLabel"] p {
        color: #374151 !important;
        font-weight: 700;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 700;
        font-size: 32px;
    }

    div[data-testid="stMetric"] * {
    color: #111827 !important;
}

div[data-testid="stMetricLabel"] {
    color: #374151 !important;
}

div[data-testid="stMetricValue"] {
    color: #111827 !important;
}

    /* ── Sidebar: dark surface, so only text drawn directly on
       that dark background gets the light color. Anything that
       renders its own light-colored box (file uploader dropzone,
       etc.) keeps dark text so it stays readable on its own bg. ── */
    section[data-testid="stSidebar"] {
        background-color: """ + SIDEBAR_BG + """;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: """ + SIDEBAR_TEXT + """;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: """ + SIDEBAR_MUTED + """;
    }
    section[data-testid="stSidebar"] hr { border-color: #334155; }

    /* File uploader drop zone: give it a dark box of its own so the
       light label text inside it stays legible, instead of light
       text landing on the widget's default light background. */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: """ + SIDEBAR_WIDGET_BG + """;
        border: 1px dashed #475569;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: """ + SIDEBAR_TEXT + """ !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
        color: """ + SIDEBAR_MUTED + """ !important;
    }
    /* "Browse files" button keeps a light surface with dark text */
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background-color: #F1F5F9 !important;
        color: """ + TEXT_DARK + """ !important;
        border: 1px solid #475569 !important;
    }

    /* Slider track numbers render on transparent bg over the dark
       sidebar, so the light color is safe there. */
    section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
        color: """ + SIDEBAR_MUTED + """;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    .footer-note {
        text-align: center;
        color: #CBD5E1;
        font-size: 0.85rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_style(ax, title, ylabel="Sales ($)", xlabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#94A3B8")
    ax.set_axisbelow(True)


# ── Hero Header ───────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📈 Sales &amp; Demand Forecasting</h1>
        <p>Superstore Sales Dataset &nbsp;·&nbsp; Time-Series Forecasting with Random Forest</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("Configure your forecast below.")
    months_ahead = st.slider("Months to Forecast", 3, 24, 12)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Superstore CSV", type=["csv"])
    st.markdown("---")

# ── Load Data ─────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("Loading and cleaning data..."):
        df_raw = load_and_clean(uploaded_file)
else:
    st.info("👆 Upload the Superstore CSV file from the sidebar to get started.")
    st.stop()

# ── Section 1: Dataset Overview ─────────────────────────────
st.markdown('<div class="section-title">🗂️ Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df_raw):,}")
col2.metric("Date Range", f"{df_raw['Order Date'].min().year} – {df_raw['Order Date'].max().year}")
col3.metric("Total Sales", f"${df_raw['Sales'].sum():,.0f}")
col4.metric("Total Profit", f"${df_raw['Profit'].sum():,.0f}")

with st.expander("Preview raw records"):
    st.dataframe(
        df_raw[['Order Date', 'Category', 'Sub-Category', 'Region', 'Sales', 'Profit']].head(20),
        use_container_width=True,
    )

# ── Section 2: Monthly Sales Trend ──────────────────────────
st.markdown('<div class="section-title">📅 Monthly Sales Trend</div>', unsafe_allow_html=True)

monthly = aggregate_monthly(df_raw)

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(monthly['Date'], monthly['Sales'], color=PRIMARY, linewidth=2)
ax1.fill_between(monthly['Date'], monthly['Sales'], alpha=0.12, color=PRIMARY)
chart_style(ax1, "Monthly Sales Over Time")
plt.tight_layout()
st.pyplot(fig1)

# ── Section 3: Model Training ───────────────────────────────
st.markdown('<div class="section-title">🤖 Model Training — Random Forest Regressor</div>', unsafe_allow_html=True)

with st.spinner("Training model..."):
    monthly_feat = build_features(monthly)
    model, train_preds, mae, rmse = train_model(monthly_feat)

col1, col2 = st.columns(2)
col1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
col2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(monthly_feat['Date'], monthly_feat['Sales'], label='Actual Sales', color=PRIMARY, linewidth=2)
ax2.plot(monthly_feat['Date'], train_preds, label='Predicted Sales', color=WARN, linestyle='--', linewidth=2)
chart_style(ax2, "Actual vs. Predicted Monthly Sales")
ax2.legend(frameon=False)
plt.tight_layout()
st.pyplot(fig2)

# ── Section 4: Future Forecast ──────────────────────────────
st.markdown(f'<div class="section-title">🔮 Sales Forecast — Next {months_ahead} Months</div>', unsafe_allow_html=True)

with st.spinner("Generating forecast..."):
    future_df = forecast_future(model, monthly_feat, months_ahead=months_ahead)

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(monthly_feat['Date'], monthly_feat['Sales'], label='Historical Sales', color=PRIMARY, linewidth=2)
ax3.plot(
    future_df['Date'], future_df['Predicted_Sales'],
    label='Forecasted Sales', color=ACCENT, linewidth=2,
    linestyle='--', marker='o', markersize=4,
)
ax3.axvline(x=monthly_feat['Date'].iloc[-1], color=MUTED, linestyle=':', linewidth=1.3, label='Forecast Start')
chart_style(ax3, f"Sales Forecast for Next {months_ahead} Months")
ax3.legend(frameon=False)
plt.tight_layout()
st.pyplot(fig3)

st.markdown("**Predicted Monthly Sales**")
forecast_display = future_df[['Date', 'Predicted_Sales']].copy()
forecast_display['Date'] = forecast_display['Date'].dt.strftime('%b %Y')
forecast_display['Predicted_Sales'] = forecast_display['Predicted_Sales'].map(lambda x: f"${x:,.2f}")
forecast_display.columns = ['Month', 'Predicted Sales']
st.dataframe(forecast_display, use_container_width=True, hide_index=True)

csv_bytes = future_df[['Date', 'Predicted_Sales']].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Forecast as CSV",
    data=csv_bytes,
    file_name="sales_forecast.csv",
    mime="text/csv",
)

# ── Footer ───────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Built for Future Interns – ML Task 1 &nbsp;·&nbsp; Dataset: Superstore Sales</div>',
    unsafe_allow_html=True,
)