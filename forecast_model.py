"""
Project FORESIGHT — Demand Forecasting Model
==========================================================================
Builds a daily demand forecast from dashboard_data.csv, backtests it
against a naive baseline using WAPE (Weighted Absolute Percentage Error),
then forecasts the next 30 days and writes prediction_results.csv for
the Streamlit dashboard to consume.

Run with:
    python forecast_model.py

Input:
    dashboard_data.csv   (same file used by streamlit_app.py)

Output:
    prediction_results.csv   (Date, Actual, Predicted, Type)
    model_performance.md     (short report: baseline vs model WAPE)
==========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
INPUT_FILE = "dashboard_data.csv"
OUTPUT_PREDICTIONS = "prediction_results.csv"
OUTPUT_REPORT = "model_performance.md"

BACKTEST_DAYS = 30      # how many recent days to hold out for testing
FORECAST_DAYS = 30      # how many future days to forecast


# --------------------------------------------------------------------
# STEP 1 — LOAD & CLEAN DATA
# --------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Place this script in the same "
            f"folder as {path} and run again."
        )

    df = pd.read_csv(path, encoding="utf-8", low_memory=False, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Drop cancelled/negative-quantity rows and rows with no date
    df = df.dropna(subset=["InvoiceDate", "Quantity"])
    df = df[df["Quantity"] > 0]

    df["Revenue"] = df["Quantity"] * df["Price"].fillna(0)
    return df


# --------------------------------------------------------------------
# STEP 2 — AGGREGATE TO DAILY DEMAND
# --------------------------------------------------------------------
def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df["InvoiceDate"].dt.date
    daily = df.groupby("Date", as_index=False)["Quantity"].sum()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)

    # Fill any missing calendar days with 0 demand so the series is continuous
    full_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(full_range).fillna(0).rename_axis("Date").reset_index()
    daily.rename(columns={"Quantity": "Demand"}, inplace=True)
    return daily


# --------------------------------------------------------------------
# STEP 3 — FEATURE ENGINEERING (for the regression model)
# --------------------------------------------------------------------
def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["day_index"] = np.arange(len(out))          # linear trend
    out["day_of_week"] = out["Date"].dt.dayofweek    # 0=Mon ... 6=Sun
    out["day_of_month"] = out["Date"].dt.day
    out["month"] = out["Date"].dt.month

    # one-hot encode day of week (captures weekly seasonality)
    dow_dummies = pd.get_dummies(out["day_of_week"], prefix="dow")
    out = pd.concat([out, dow_dummies], axis=1)
    return out


# --------------------------------------------------------------------
# STEP 4 — WAPE METRIC
# --------------------------------------------------------------------
def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — robust to zero-demand days."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denom * 100)


# --------------------------------------------------------------------
# STEP 5 — BASELINE MODEL (seasonal naive: same weekday, previous week)
# --------------------------------------------------------------------
def seasonal_naive_forecast(train: pd.DataFrame, test_len: int) -> np.ndarray:
    last_week = train["Demand"].values[-7:]
    if len(last_week) < 7:
        # not enough history — fall back to overall mean
        return np.full(test_len, train["Demand"].mean())
    reps = int(np.ceil(test_len / 7))
    return np.tile(last_week, reps)[:test_len]


# --------------------------------------------------------------------
# STEP 6 — MAIN PIPELINE
# --------------------------------------------------------------------
def main():
    print("📦 Project FORESIGHT — Forecast Model")
    print("=" * 60)

    # ---- Load & prepare ----
    df = load_data(INPUT_FILE)
    daily = build_daily_series(df)
    print(f"✅ Loaded {len(df):,} transactions -> {len(daily):,} daily demand points "
          f"({daily['Date'].min().date()} to {daily['Date'].max().date()})")

    if len(daily) < BACKTEST_DAYS + 14:
        raise ValueError(
            f"Not enough history to backtest ({len(daily)} days available). "
            f"Need at least {BACKTEST_DAYS + 14} days of data."
        )

    featured = add_features(daily)
    feature_cols = [c for c in featured.columns if c.startswith("dow_")] + ["day_index", "day_of_month", "month"]

    # ---- Train/Test split (last BACKTEST_DAYS held out) ----
    train = featured.iloc[: -BACKTEST_DAYS].reset_index(drop=True)
    test = featured.iloc[-BACKTEST_DAYS:].reset_index(drop=True)

    # ---- Baseline: seasonal naive ----
    baseline_preds = seasonal_naive_forecast(train, len(test))
    baseline_wape = wape(test["Demand"].values, baseline_preds)

    # ---- Model: Linear Regression with trend + weekly seasonality ----
    model = LinearRegression()
    model.fit(train[feature_cols], train["Demand"])
    model_test_preds = model.predict(test[feature_cols])
    model_test_preds = np.clip(model_test_preds, a_min=0, a_max=None)  # demand can't be negative
    model_wape = wape(test["Demand"].values, model_test_preds)

    improvement = baseline_wape - model_wape
    print(f"\n📊 Backtest Results (last {BACKTEST_DAYS} days held out):")
    print(f"   Baseline (Seasonal Naive) WAPE : {baseline_wape:.2f}%")
    print(f"   Model    (Linear Regression) WAPE : {model_wape:.2f}%")
    if model_wape < baseline_wape:
        print(f"   ✅ Model beats baseline by {improvement:.2f} percentage points")
    else:
        print(f"   ⚠️ Model did NOT beat baseline (diff {improvement:.2f} pts) — "
              f"consider more history or a different model")

    # ---- Retrain on FULL data, forecast future FORECAST_DAYS ----
    model_full = LinearRegression()
    model_full.fit(featured[feature_cols], featured["Demand"])

    last_date = featured["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
    future_df = pd.DataFrame({"Date": future_dates})
    future_df["day_index"] = np.arange(len(featured), len(featured) + FORECAST_DAYS)
    future_df["day_of_week"] = future_df["Date"].dt.dayofweek
    future_df["day_of_month"] = future_df["Date"].dt.day
    future_df["month"] = future_df["Date"].dt.month
    dow_dummies_future = pd.get_dummies(future_df["day_of_week"], prefix="dow")
    future_df = pd.concat([future_df, dow_dummies_future], axis=1)

    # ensure future_df has every dow_ column the model was trained on
    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    future_preds = model_full.predict(future_df[feature_cols])
    future_preds = np.clip(future_preds, a_min=0, a_max=None)

    # ---- Assemble output: backtest period (Actual+Predicted) + future (Predicted only) ----
    backtest_out = pd.DataFrame({
        "Date": test["Date"].values,
        "Actual": test["Demand"].values.round(1),
        "Predicted": model_test_preds.round(1),
        "Type": "Backtest",
    })
    future_out = pd.DataFrame({
        "Date": future_df["Date"].values,
        "Actual": np.nan,
        "Predicted": future_preds.round(1),
        "Type": "Forecast",
    })
    result = pd.concat([backtest_out, future_out], ignore_index=True)
    result.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\n✅ Saved predictions -> {OUTPUT_PREDICTIONS} ({len(result)} rows)")

    # ---- Write a short performance report ----
    with open(OUTPUT_REPORT, "w") as f:
        f.write("# Project FORESIGHT — Model Performance Report\n\n")
        f.write(f"**Data range:** {daily['Date'].min().date()} to {daily['Date'].max().date()} "
                f"({len(daily)} days)\n\n")
        f.write(f"**Backtest window:** last {BACKTEST_DAYS} days\n\n")
        f.write("## Results\n\n")
        f.write("| Model | WAPE |\n|---|---|\n")
        f.write(f"| Baseline (Seasonal Naive) | {baseline_wape:.2f}% |\n")
        f.write(f"| Linear Regression (trend + weekly seasonality) | {model_wape:.2f}% |\n\n")
        if model_wape < baseline_wape:
            f.write(f"**Conclusion:** The model beats the seasonal-naive baseline by "
                     f"**{improvement:.2f} percentage points**, indicating the trend and "
                     f"weekly-seasonality features add real forecasting value.\n\n")
        else:
            f.write(f"**Conclusion:** The model did not outperform the baseline in this run "
                     f"(diff {improvement:.2f} pts). Consider adding more historical data, "
                     f"holiday effects, or product-level features.\n\n")
        f.write(f"**Forecast horizon:** next {FORECAST_DAYS} days "
                 f"({future_dates[0].date()} to {future_dates[-1].date()})\n")
    print(f"✅ Saved report -> {OUTPUT_REPORT}")
    print("\n🎉 Done. Restart your Streamlit app to see the updated forecast.")


if __name__ == "__main__":
    main()