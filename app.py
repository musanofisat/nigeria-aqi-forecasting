"""
Nigeria Air Quality Forecast — Streamlit Dashboard
====================================================
Interactive front-end for the capstone forecasting pipeline (see forecasting.py).
Lets a user pick National / a State / a City, trains & compares 6 models on the
fly, and shows a 12-month-ahead AQI forecast with confidence intervals.

Run locally:    streamlit run app.py
Deploy:         see README.md for Streamlit Community Cloud steps
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import forecasting as fc

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Nigeria AQI Forecast",
    page_icon="🌍",
    layout="wide",
)

DATA_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "data", "nigeria_air_quality_2014_2025.csv")

SEVERITY_COLORS = {
    "Good": "#00A651",
    "Moderate": "#FFD700",
    "Unhealthy for Sensitive Groups": "#FF8C00",
    "Unhealthy": "#E53935",
    "Very Unhealthy": "#8E24AA",
    "Hazardous": "#6D2932",
}


# ----------------------------------------------------------------------
# Cached data / compute layers
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_load_data(path_or_buffer) -> pd.DataFrame:
    return fc.load_data(path_or_buffer)


@st.cache_data(show_spinner=False)
def cached_geo_options(df: pd.DataFrame) -> dict:
    return fc.get_geo_options(df)


@st.cache_resource(show_spinner=False)
def cached_pipeline(df_hash: str, _df: pd.DataFrame, level: str, value, horizon: int) -> dict:
    """
    _df is prefixed with underscore so Streamlit doesn't try to hash the whole
    DataFrame; df_hash (a cheap fingerprint) plus level/value/horizon key the cache.
    """
    return fc.run_pipeline(_df, level, value, horizon=horizon)


# ----------------------------------------------------------------------
# Sidebar — data source & selection controls
# ----------------------------------------------------------------------

st.sidebar.title("🌍 Nigeria AQI Forecast")
st.sidebar.caption("12-year monitoring data (2014–2025) · 72 cities · 37 states")

uploaded = st.sidebar.file_uploader(
    "Optional: upload an updated CSV (same schema) to retrain on newer data",
    type=["csv"],
)

with st.spinner("Loading data..."):
    df = cached_load_data(uploaded if uploaded is not None else DATA_PATH_DEFAULT)
    geo = cached_geo_options(df)

st.sidebar.markdown("---")
level = st.sidebar.radio("Forecast level", ["National", "State", "City"], index=0)

value = None
if level == "State":
    value = st.sidebar.selectbox("Select a state", geo["states"])
elif level == "City":
    state_filter = st.sidebar.selectbox("Filter by state (optional)", ["All"] + geo["states"])
    city_choices = geo["all_cities"] if state_filter == "All" else geo["city_by_state"][state_filter]
    value = st.sidebar.selectbox("Select a city", city_choices)

horizon = st.sidebar.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Methodology: Naive & Seasonal-Naive baselines, Holt-Winters, auto-tuned SARIMA, "
    "Random Forest and XGBoost (lag + rolling + calendar + pollutant features) are all "
    "trained and evaluated on a held-out window; the lowest-RMSE model is retrained on "
    "full history to produce the forecast shown."
)

# ----------------------------------------------------------------------
# Run pipeline (cached per level/value/horizon)
# ----------------------------------------------------------------------

df_fingerprint = f"{len(df)}-{df['Date'].max()}"
label = "Nigeria (National Average)" if level == "National" else value

with st.spinner(f"Training & evaluating models for {label}..."):
    try:
        result = cached_pipeline(df_fingerprint, df, level, value, horizon)
    except ValueError as e:
        st.error(str(e))
        st.stop()

agg = result["agg"]
comparison = result["comparison"]
forecast_table = result["forecast_table"]
decomposition = result["decomposition"]
best_model = comparison["best_model"]
best_row = comparison["results_df"].iloc[0]

# ----------------------------------------------------------------------
# Header + headline metrics
# ----------------------------------------------------------------------

st.title(f"Air Quality Forecast — {label}")
st.caption(f"Forecast horizon: {horizon} months ahead, starting {forecast_table.index[0].strftime('%B %Y')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Winning model", best_model)
c2.metric("Test RMSE", f"{best_row['RMSE']:.2f}")
c3.metric("Test MAPE", f"{best_row['MAPE (%)']:.1f}%")
worst_month = forecast_table["Forecast_AQI"].idxmax()
c4.metric("Peak forecast month", worst_month.strftime("%b %Y"),
          f"AQI {forecast_table['Forecast_AQI'].max():.0f}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

tab_forecast, tab_history, tab_models, tab_about = st.tabs(
    ["📈 Forecast", "📊 Historical Trends", "🧪 Model Comparison", "ℹ️ About"]
)

# ---- Forecast tab ----
with tab_forecast:
    # ----------------------------------------------------------
    # Forecast Insight
    # ----------------------------------------------------------
    highest = forecast_table["Forecast_AQI"].idxmax()
    lowest = forecast_table["Forecast_AQI"].idxmin()

    highest_aqi = forecast_table.loc[highest, "Forecast_AQI"]
    lowest_aqi = forecast_table.loc[lowest, "Forecast_AQI"]

    highest_category = forecast_table.loc[highest, "Severity"]
    lowest_category = forecast_table.loc[lowest, "Severity"]

    st.subheader("🔮 Forecast Insight")

    st.info(
        f"Air quality is forecast to be highest in "
        f"**{highest.strftime('%B %Y')}**, with an AQI of "
        f"**{highest_aqi:.1f} ({highest_category})**. "
        f"The lowest forecast is expected in "
        f"**{lowest.strftime('%B %Y')}**, with an AQI of "
        f"**{lowest_aqi:.1f} ({lowest_category})**."
    )

    # ----------------------------------------------------------
    # Public-health interpretation
    # ----------------------------------------------------------
    public_health_messages = {
        "Good": "Air quality is generally considered satisfactory. No special precautions are expected for the general population.",
        "Moderate": "Air quality may affect unusually sensitive people. People who are particularly sensitive to air pollution can consider reducing prolonged outdoor activity if they notice symptoms.",
        "Unhealthy for Sensitive Groups": "Sensitive groups may experience health effects. Schools, healthcare facilities, and vulnerable individuals can use this period to plan activities and reduce prolonged exposure when appropriate.",
        "Unhealthy": "More people may experience health effects. Consider reducing prolonged outdoor exposure, especially for sensitive groups, and use the forecast to support preparedness and environmental-health planning.",
        "Very Unhealthy": "The risk of health effects is increased. Communities, schools, and healthcare facilities can use this period for heightened preparedness and consider limiting prolonged outdoor activities.",
        "Hazardous": "Air pollution is forecast at a hazardous level. This warrants strong public-health awareness and preparedness, with vulnerable people taking particular care to minimize prolonged exposure."
    }

    insight_message = public_health_messages.get(
        highest_category,
        "Use the forecast as an early-warning signal for air-quality planning and preparedness."
    )

    st.subheader("🩺 What this means for public health")
    st.warning(insight_message)
    st.caption(
        "These are general public-health considerations, not medical advice. "
        "Forecasts are predictions and may differ from actual conditions."
    )

    hist_tail = agg["AQI"].iloc[-24:]
    hist_df = pd.DataFrame({"Date": hist_tail.index, "AQI": hist_tail.values, "Type": "Historical"})
    fut_df = pd.DataFrame({
        "Date": forecast_table.index, "AQI": forecast_table["Forecast_AQI"].values, "Type": "Forecast"
    })
    band_df = forecast_table.reset_index().rename(columns={"index": "Date"})

    base = alt.Chart(pd.concat([hist_df, fut_df])).encode(x="Date:T")
    line = base.mark_line(point=True).encode(
        y=alt.Y("AQI:Q", title="AQI"),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Historical", "Forecast"],
                                                    range=["#2C3E50", "#E53935"])),
    )
    band = alt.Chart(band_df).mark_area(opacity=0.15, color="#E53935").encode(
        x="Date:T", y="Lower_95%_CI:Q", y2="Upper_95%_CI:Q"
    )
    st.altair_chart((band + line).properties(height=420), width='stretch')

    st.subheader("Forecast Table")
    display_table = forecast_table.copy()
    display_table.index = display_table.index.strftime("%b %Y")
    display_table.index.name = "Month"

    def _color_severity(val):
        color = SEVERITY_COLORS.get(val, "#ffffff")
        return f"background-color: {color}22; color: {color}; font-weight: 600"

    st.dataframe(
        display_table.style.map(_color_severity, subset=["Severity"]),
        width='stretch',
    )

    st.download_button(
        "⬇ Download forecast as CSV",
        data=display_table.to_csv().encode("utf-8"),
        file_name=f"aqi_forecast_{level.lower()}_{(value or 'nigeria').replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---- Historical trends tab ----
with tab_history:
    st.subheader("Full Historical AQI Series")
    hist_full = agg["AQI"].reset_index()
    hist_full.columns = ["Date", "AQI"]
    chart = alt.Chart(hist_full).mark_line(color="#2C3E50").encode(x="Date:T", y="AQI:Q")
    st.altair_chart(chart.properties(height=350), width='stretch')

    st.subheader("Seasonal Decomposition")
    dcomp_df = pd.DataFrame({
        "Date": decomposition.observed.index,
        "Trend": decomposition.trend.values,
        "Seasonal": decomposition.seasonal.values,
        "Residual": decomposition.resid.values,
    }).melt("Date", var_name="Component", value_name="Value")
    dcomp_chart = alt.Chart(dcomp_df).mark_line().encode(
        x="Date:T", y="Value:Q", color="Component:N",
        row=alt.Row("Component:N", header=alt.Header(labelAngle=0)),
    ).resolve_scale(y="independent").properties(height=130)
    st.altair_chart(dcomp_chart, width='stretch')

    st.subheader("Average AQI by Calendar Month (Seasonality)")
    month_avg = agg.copy()
    month_avg["Month"] = month_avg.index.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = month_avg.groupby("Month")["AQI"].mean().reindex(month_order).reset_index()
    bar = alt.Chart(monthly).mark_bar(color="#3E4A89").encode(
        x=alt.X("Month:N", sort=month_order), y="AQI:Q"
    )
    st.altair_chart(bar.properties(height=300), width='stretch')

# ---- Model comparison tab ----
with tab_models:
    st.subheader(f"Held-out Test Performance ({horizon}-month window)")
    st.dataframe(comparison["results_df"], width='stretch')
    st.caption("Lower RMSE/MAE/MAPE = better. The lowest-RMSE model above is retrained on the "
               "full series to produce the forecast on the Forecast tab.")

    st.subheader("Test-Period: Actual vs. Each Model's Prediction")
    test = comparison["test"]
    plot_df = pd.DataFrame({"Date": test.index, "Actual": test.values})
    for name, series in comparison["forecasts"].items():
        plot_df[name] = series.values
    plot_long = plot_df.melt("Date", var_name="Series", value_name="AQI")
    chart2 = alt.Chart(plot_long).mark_line(point=True).encode(
        x="Date:T", y="AQI:Q", color="Series:N",
        strokeDash=alt.condition(alt.datum.Series == "Actual", alt.value([1, 0]), alt.value([4, 2])),
    )
    st.altair_chart(chart2.properties(height=400), width='stretch')

# ---- About tab ----
with tab_about:
    st.markdown(f"""
    ### About this dashboard
    This app is the deployed version of the *Nigeria AQI Forecasting* capstone notebook.
    For the selected geography, it:

    1. Aggregates monthly AQI (and pollutant/environmental drivers) from the underlying dataset
    2. Trains and evaluates **6 candidate models** on a held-out {horizon}-month window:
       Naive, Seasonal Naive, Holt-Winters, auto-tuned SARIMA, Random Forest, and XGBoost
    3. Automatically selects the model with the **lowest RMSE** on that held-out window
    4. Retrains the winner on the **full history** and forecasts {horizon} months forward,
       with 95% confidence intervals derived from test-set residuals (tree models / Holt-Winters)
       or native statistical intervals (SARIMA)

    **Currently viewing:** {label} · **{len(agg)} months** of history
    ({agg.index.min().strftime('%b %Y')} – {agg.index.max().strftime('%b %Y')})

    **Data freshness:** to refresh with new monitoring data, upload an updated CSV
    (same column schema) using the sidebar uploader — no code changes needed.

    ---
    Built with Streamlit · scikit-learn · XGBoost · statsmodels · pmdarima
    """)
