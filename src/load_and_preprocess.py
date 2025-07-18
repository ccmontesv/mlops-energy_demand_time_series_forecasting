# load_and_preprocess.py

import pandas as pd
import numpy as np
import os

def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV and set datetime index."""
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess time series data: feature engineering, lag/rolling, imputation."""
    
    # Select relevant columns
    cols_to_keep = [
        "HU_solar_generation_actual",
        "DE_LU_load_forecast_entsoe_transparency",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual",
        "DE_LU_wind_onshore_generation_actual",
        "DE_LU_wind_offshore_generation_actual",
        "AT_price_day_ahead",
        "DE_solar_profile",
        "DE_wind_profile",
        "DE_solar_capacity",
        "DE_wind_capacity",
        'DE_solar_generation_actual',
        'DE_wind_generation_actual',
        'DE_load_actual_entsoe_transparency'
    ]
    df = df[cols_to_keep]

    # Time-based features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(0, 6).astype(int)

    # Drop columns with excessive missing values
    missing_pct = df.isna().mean() * 100
    df = df.drop(columns=missing_pct[missing_pct > 65].index)

    # Interpolation for known integer columns
    cols_int = [
        "AT_price_day_ahead", "DE_solar_profile", "DE_wind_profile",
        "DE_solar_capacity", "DE_wind_capacity"
    ]
    df[cols_int] = df[cols_int].interpolate().fillna(method='ffill').fillna(method='bfill')

    # Lag features
    target = 'DE_load_actual_entsoe_transparency'
    df['lag_15min'] = df[target].shift(1)
    df['lag_1hour'] = df[target].shift(4)
    df['lag_1day'] = df[target].shift(96)
    df['lag_1week'] = df[target].shift(96 * 7)

    # Rolling window features (std)
    df['rolling_std_1hour'] = df[target].rolling(window=4).std()
    df['rolling_std_6hour'] = df[target].rolling(window=24).std()
    df['rolling_std_1day'] = df[target].rolling(window=96).std()
    df['rolling_std_1week'] = df[target].rolling(window=96*7).std()

    return df

def get_clean_data(df: pd.DataFrame, target: str) -> tuple:
    """Remove rows with missing target values and return features/target split."""
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    features = [col for col in df_clean.columns if col != target]
    X = df_clean[features]
    y = df_clean[target]
    return X, y

def save_processed_data(df: pd.DataFrame, filename: str = "processed_data.csv"):
    """Save the preprocessed DataFrame to data/processed/."""
    os.makedirs("data/processed", exist_ok=True)
    output_path = os.path.join("data/processed", filename)
    df.to_csv(output_path)
    print(f" Processed data saved to {output_path}")