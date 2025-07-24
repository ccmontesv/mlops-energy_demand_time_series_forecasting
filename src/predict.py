import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.load_and_preprocess import preprocess_data

# === CONFIGURATION ===
TARGET_COL = 'DE_load_actual_entsoe_transparency'
MODEL_PATH = Path("models/xgb_model.pkl")
FEATURES_PATH = Path("models/feature_names.pkl")
RAW_DATA_PATH = Path("data/raw/time_series_15min_singleindex.csv")
FORECAST_OUTPUT_PATH = Path("outputs/forecast_next_day.csv")

FREQ = "15min"
STEPS_AHEAD = 96          # 1 full day of 15-min steps
HISTORY_DAYS = 8          # for lag/rolling window support


def load_artifacts():
    """Load model and feature names."""
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


def load_history(raw_data_path: Path, history_days: int) -> pd.DataFrame:
    """Load and filter recent history to support lag/rolling features."""
    df = pd.read_csv(raw_data_path, parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp").sort_index()

    end_time = df.index.max()
    start_time = end_time - pd.Timedelta(days=history_days)
    return df.loc[start_time:]


def generate_forecast_timestamps(last_time: pd.Timestamp, steps: int, freq: str) -> pd.DatetimeIndex:
    """Create forecast horizon index."""
    return pd.date_range(start=last_time + pd.Timedelta(freq), periods=steps, freq=freq)


def forecast_next_day(df_history: pd.DataFrame, model, feature_names: list) -> pd.DataFrame:
    """Generate 96-step recursive forecast using lag-based features."""
    last_time = df_history.index.max()
    future_index = generate_forecast_timestamps(last_time, STEPS_AHEAD, FREQ)

    predictions = []

    for timestamp in future_index:
        # Combine true history and past predictions
        combined_df = pd.concat([df_history, pd.DataFrame(predictions)], axis=0)
        combined_df = preprocess_data(combined_df)

        try:
            row = combined_df.loc[[timestamp]]
            X = row[feature_names]
        except KeyError:
            print(f"Skipping {timestamp} — not enough history to compute features.")
            continue

        pred = model.predict(X)[0]
        predictions.append(pd.DataFrame({TARGET_COL: [pred]}, index=[timestamp]))

    forecast_df = pd.concat(predictions)
    forecast_df.columns = ["predicted_load"]
    forecast_df.index.name = "utc_timestamp"
    return forecast_df


def main():
    print("Loading artifacts and data...")
    model, feature_names = load_artifacts()
    df_history = load_history(RAW_DATA_PATH, history_days=HISTORY_DAYS)

    print("Generating forecast...")
    forecast_df = forecast_next_day(df_history, model, feature_names)

    print(f"Saving forecast to {FORECAST_OUTPUT_PATH}")
    forecast_df.to_csv(FORECAST_OUTPUT_PATH)
    print("Forecast complete and saved.")


if __name__ == "__main__":
    main()

