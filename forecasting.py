"""
forecasting.py
================
Core forecasting engine for the Nigeria Air Quality (AQI) project.

Deliberately kept free of any Streamlit / UI code so it can be:
  - imported and unit-tested with plain Python
  - reused by the Streamlit dashboard (app.py)
  - reused by a batch retraining job / API, if you build one later

Methodology mirrors the capstone notebook: baselines -> Holt-Winters -> SARIMA
-> Random Forest / XGBoost, evaluated on a held-out horizon, with the winning
model retrained on full history to produce the final forecast.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

RANDOM_STATE = 42
DEFAULT_HORIZON = 12

EXO_COLS = {
    "PM25": "PM2.5 (ug/m3)",
    "PM10": "PM10 (ug/m3)",
    "Deforestation": "Deforestation_Rate_%",
    "Industry_Growth": "Industry_Growth_%",
}


# ----------------------------------------------------------------------
# Data loading & selection helpers
# ----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load and lightly validate the raw AQI dataset."""
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"State", "City", "Date", "AQI", *EXO_COLS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def get_geo_options(df: pd.DataFrame) -> dict:
    """Return sorted state/city lists for populating dashboard dropdowns."""
    states = sorted(df["State"].unique().tolist())
    city_by_state = (
        df.groupby("State")["City"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )
    all_cities = sorted(df["City"].unique().tolist())
    return {"states": states, "city_by_state": city_by_state, "all_cities": all_cities}


def filter_subset(df: pd.DataFrame, level: str, value: str | None) -> pd.DataFrame:
    """level in {'National', 'State', 'City'}."""
    if level == "National" or value is None:
        return df
    if level == "State":
        return df[df["State"] == value]
    if level == "City":
        return df[df["City"] == value]
    raise ValueError(f"Unknown level: {level}")


# ----------------------------------------------------------------------
# Series construction & feature engineering
# ----------------------------------------------------------------------

def build_monthly_series(subset: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a (national/state/city) subset into one monthly AQI + exogenous series."""
    agg = subset.groupby("Date").agg(
        AQI=("AQI", "mean"),
        PM25=(EXO_COLS["PM25"], "mean"),
        PM10=(EXO_COLS["PM10"], "mean"),
        Deforestation=(EXO_COLS["Deforestation"], "mean"),
        Industry_Growth=(EXO_COLS["Industry_Growth"], "mean"),
    )
    agg = agg.asfreq("MS")
    # In case a specific city/state slice ever has a gap, interpolate rather than crash.
    if agg.isna().any().any():
        agg = agg.interpolate(limit_direction="both")
    return agg


def build_features(agg: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Lag / rolling / calendar feature engineering, identical to the notebook."""
    feat_df = agg.copy()

    for lag in [1, 2, 3, 6, 12]:
        feat_df[f"AQI_lag{lag}"] = feat_df["AQI"].shift(lag)

    feat_df["AQI_roll3"] = feat_df["AQI"].shift(1).rolling(3).mean()
    feat_df["AQI_roll12"] = feat_df["AQI"].shift(1).rolling(12).mean()

    feat_df["month"] = feat_df.index.month
    feat_df["month_sin"] = np.sin(2 * np.pi * feat_df["month"] / 12)
    feat_df["month_cos"] = np.cos(2 * np.pi * feat_df["month"] / 12)

    feat_df = feat_df.dropna()
    feature_cols = [c for c in feat_df.columns if c not in ["AQI", "month"]]
    return feat_df, feature_cols


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / np.clip(actual, 1e-6, None))) * 100
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def classify_aqi(value: float) -> str:
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Moderate"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif value <= 200:
        return "Unhealthy"
    elif value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


# ----------------------------------------------------------------------
# Model training + test-set evaluation
# ----------------------------------------------------------------------

def run_model_comparison(agg: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    """
    Trains every candidate model on all-but-last-`horizon` months, evaluates on the
    held-out months, and returns everything the dashboard needs to render the
    comparison charts/tables.
    """
    ts = agg["AQI"]
    if len(ts) <= horizon + 12:
        raise ValueError(
            f"Series too short ({len(ts)} months) for a {horizon}-month evaluation "
            f"with seasonal (12-month) features. Need at least {horizon + 13} months."
        )

    train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]
    results = []
    forecasts = {}
    fitted = {}  # keep fitted objects around in case caller wants them (e.g. SARIMA orders)

    # --- Baselines ---
    naive_pred = pd.Series(np.repeat(train.iloc[-1], horizon), index=test.index)
    results.append(evaluate(test.values, naive_pred.values, "Naive (last value)"))
    forecasts["Naive"] = naive_pred

    seasonal_naive_pred = pd.Series(train.iloc[-12:].values, index=test.index)
    results.append(evaluate(test.values, seasonal_naive_pred.values, "Seasonal Naive (t-12)"))
    forecasts["Seasonal Naive"] = seasonal_naive_pred

    # --- Holt-Winters ---
    try:
        hw_model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
        ).fit()
        hw_pred = hw_model.forecast(horizon)
        results.append(evaluate(test.values, hw_pred.values, "Holt-Winters"))
        forecasts["Holt-Winters"] = hw_pred
        fitted["hw_test_resid_std"] = np.std(hw_pred.values - test.values, ddof=1)
    except Exception:
        pass

    # --- SARIMA (auto_arima, constrained search for interactive speed) ---
    try:
        sarima_auto = pm.auto_arima(
            train, seasonal=True, m=12,
            start_p=0, start_q=0, max_p=2, max_q=2,
            start_P=0, start_Q=0, max_P=1, max_Q=1,
            d=None, D=None, stepwise=True,
            suppress_warnings=True, error_action="ignore",
        )
        sarima_pred = pd.Series(sarima_auto.predict(n_periods=horizon), index=test.index)
        model_label = f"SARIMA{sarima_auto.order}x{sarima_auto.seasonal_order}"
        results.append(evaluate(test.values, sarima_pred.values, model_label))
        forecasts[model_label] = sarima_pred
        fitted["sarima_order"] = sarima_auto.order
        fitted["sarima_seasonal_order"] = sarima_auto.seasonal_order
    except Exception:
        pass

    # --- ML models (Random Forest / XGBoost) ---
    feat_df, feature_cols = build_features(agg)
    if len(feat_df) > horizon + 5:
        X, y = feat_df[feature_cols], feat_df["AQI"]
        X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
        y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

        rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2,
                                    random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        rf_pred = pd.Series(rf.predict(X_test), index=y_test.index)
        results.append(evaluate(y_test.values, rf_pred.values, "Random Forest"))
        forecasts["Random Forest"] = rf_pred
        fitted["rf_model"] = rf
        fitted["rf_test_resid_std"] = np.std(rf_pred.values - y_test.values, ddof=1)

        xgb_model = xgb.XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=RANDOM_STATE)
        xgb_model.fit(X_train, y_train)
        xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)
        results.append(evaluate(y_test.values, xgb_pred.values, "XGBoost"))
        forecasts["XGBoost"] = xgb_pred
        fitted["xgb_model"] = xgb_model
        fitted["xgb_test_resid_std"] = np.std(xgb_pred.values - y_test.values, ddof=1)
        fitted["feature_cols"] = feature_cols
        fitted["feat_df"] = feat_df

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return {
        "ts": ts, "train": train, "test": test,
        "results_df": results_df, "forecasts": forecasts, "fitted": fitted,
        "best_model": results_df.iloc[0]["Model"],
    }


# ----------------------------------------------------------------------
# Final forecast (retrain winner on full history, project forward)
# ----------------------------------------------------------------------

def _recursive_tree_forecast(model, ts: pd.Series, agg: pd.DataFrame, feature_cols: list[str],
                              future_index: pd.DatetimeIndex, resid_std: float) -> pd.DataFrame:
    exo_seasonal_avg = agg.groupby(agg.index.month)[["PM25", "PM10", "Deforestation", "Industry_Growth"]].mean()
    aqi_history = list(ts.values)
    preds = []

    for date in future_index:
        m = date.month
        row = {
            "PM25": exo_seasonal_avg.loc[m, "PM25"],
            "PM10": exo_seasonal_avg.loc[m, "PM10"],
            "Deforestation": exo_seasonal_avg.loc[m, "Deforestation"],
            "Industry_Growth": exo_seasonal_avg.loc[m, "Industry_Growth"],
            "AQI_lag1": aqi_history[-1], "AQI_lag2": aqi_history[-2],
            "AQI_lag3": aqi_history[-3], "AQI_lag6": aqi_history[-6],
            "AQI_lag12": aqi_history[-12],
            "AQI_roll3": np.mean(aqi_history[-3:]), "AQI_roll12": np.mean(aqi_history[-12:]),
            "month_sin": np.sin(2 * np.pi * m / 12), "month_cos": np.cos(2 * np.pi * m / 12),
        }
        X_step = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X_step)[0])
        preds.append(pred)
        aqi_history.append(pred)

    forecast = pd.Series(preds, index=future_index, name="Forecast_AQI")
    return pd.DataFrame({
        "Forecast_AQI": forecast,
        "Lower_95%_CI": forecast - 1.96 * resid_std,
        "Upper_95%_CI": forecast + 1.96 * resid_std,
    })


def forecast_future(agg: pd.DataFrame, comparison: dict, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """
    Retrains whichever model won `run_model_comparison` on the FULL series and
    produces the forward-looking forecast table (with 95% CI + severity labels).
    """
    ts = agg["AQI"]
    best_model = comparison["best_model"]
    fitted = comparison["fitted"]
    future_index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    if best_model.startswith("SARIMA"):
        final_model = pm.ARIMA(order=fitted["sarima_order"],
                                seasonal_order=fitted["sarima_seasonal_order"],
                                suppress_warnings=True).fit(ts)
        point_fc, ci = final_model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        table = pd.DataFrame({
            "Forecast_AQI": pd.Series(point_fc, index=future_index),
            "Lower_95%_CI": pd.Series(ci[:, 0], index=future_index),
            "Upper_95%_CI": pd.Series(ci[:, 1], index=future_index),
        })

    elif best_model == "Holt-Winters":
        final_model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12,
                                            damped_trend=True).fit()
        point_fc = final_model.forecast(horizon)
        point_fc.index = future_index
        std = fitted.get("hw_test_resid_std", point_fc.std() * 0.1)
        table = pd.DataFrame({
            "Forecast_AQI": point_fc,
            "Lower_95%_CI": point_fc - 1.96 * std,
            "Upper_95%_CI": point_fc + 1.96 * std,
        })

    elif best_model in ("Random Forest", "XGBoost"):
        model = fitted["rf_model"] if best_model == "Random Forest" else fitted["xgb_model"]
        std = fitted["rf_test_resid_std"] if best_model == "Random Forest" else fitted["xgb_test_resid_std"]
        table = _recursive_tree_forecast(model, ts, agg, fitted["feature_cols"], future_index, std)

    else:  # Naive / Seasonal Naive fallback (only wins on pathological tiny series)
        seasonal_base = ts.iloc[-12:].values
        reps = int(np.ceil(horizon / 12))
        point_fc = pd.Series(np.tile(seasonal_base, reps)[:horizon], index=future_index)
        std = ts.std()
        table = pd.DataFrame({
            "Forecast_AQI": point_fc,
            "Lower_95%_CI": point_fc - 1.96 * std,
            "Upper_95%_CI": point_fc + 1.96 * std,
        })

    table["Forecast_AQI"] = table["Forecast_AQI"].round(1)
    table["Lower_95%_CI"] = table["Lower_95%_CI"].round(1)
    table["Upper_95%_CI"] = table["Upper_95%_CI"].round(1)
    table["Severity"] = table["Forecast_AQI"].apply(classify_aqi)
    return table


def get_decomposition(agg: pd.DataFrame):
    """Return a statsmodels DecomposeResult for the trend/seasonal/residual chart."""
    return seasonal_decompose(agg["AQI"], model="additive", period=12)


# ----------------------------------------------------------------------
# Lightweight dashboard forecast
# ----------------------------------------------------------------------

def run_fast_pipeline(df: pd.DataFrame, level: str, value: str | None,
                      horizon: int = DEFAULT_HORIZON) -> dict:
    """
    Lightweight forecasting path for the public dashboard.

    The full multi-model comparison remains available in run_pipeline() for
    research/evaluation. The dashboard uses a fixed seasonal SARIMA configuration
    so changing State/City does not trigger an expensive auto-model search.
    """
    subset = filter_subset(df, level, value)
    agg = build_monthly_series(subset)

    ts = agg["AQI"]
    if len(ts) < 25:
        raise ValueError("Not enough monthly history to generate a seasonal forecast.")

    future_index = pd.date_range(
        ts.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    try:
        model = pm.ARIMA(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            suppress_warnings=True,
        ).fit(ts)

        point_fc, ci = model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=0.05,
        )

        forecast_table = pd.DataFrame({
            "Forecast_AQI": pd.Series(point_fc, index=future_index),
            "Lower_95%_CI": pd.Series(ci[:, 0], index=future_index),
            "Upper_95%_CI": pd.Series(ci[:, 1], index=future_index),
        })
    except Exception:
        # Robust fallback for an unusual local series.
        seasonal_base = ts.iloc[-12:].values
        reps = int(np.ceil(horizon / 12))
        point_fc = pd.Series(
            np.tile(seasonal_base, reps)[:horizon],
            index=future_index,
        )
        std = max(float(ts.std()), 1.0)
        forecast_table = pd.DataFrame({
            "Forecast_AQI": point_fc,
            "Lower_95%_CI": point_fc - 1.96 * std,
            "Upper_95%_CI": point_fc + 1.96 * std,
        })

    forecast_table["Forecast_AQI"] = forecast_table["Forecast_AQI"].clip(lower=0).round(1)
    forecast_table["Lower_95%_CI"] = forecast_table["Lower_95%_CI"].clip(lower=0).round(1)
    forecast_table["Upper_95%_CI"] = forecast_table["Upper_95%_CI"].clip(lower=0).round(1)
    forecast_table["Severity"] = forecast_table["Forecast_AQI"].apply(classify_aqi)

    return {
        "agg": agg,
        "forecast_table": forecast_table,
        "decomposition": get_decomposition(agg),
    }


# ----------------------------------------------------------------------
# One-call convenience wrapper used by the dashboard
# ----------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, level: str, value: str | None,
                  horizon: int = DEFAULT_HORIZON) -> dict:
    subset = filter_subset(df, level, value)
    agg = build_monthly_series(subset)
    comparison = run_model_comparison(agg, horizon=horizon)
    forecast_table = forecast_future(agg, comparison, horizon=horizon)
    decomposition = get_decomposition(agg)
    return {
        "agg": agg,
        "comparison": comparison,
        "forecast_table": forecast_table,
        "decomposition": decomposition,
    }
