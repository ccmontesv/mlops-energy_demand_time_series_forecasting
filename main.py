# main.py

import joblib
import os
from src.load_and_preprocess import load_data, preprocess_data, get_clean_data, save_processed_data
from src.train_model import train_and_optimize

def main():
    # Use relative path instead of hardcoded absolute path
    path = "data/raw/time_series_15min_singleindex.csv"
    target = 'DE_load_actual_entsoe_transparency'

    print("Loading and preprocessing data...")
    df = load_data(path)
    df = preprocess_data(df)
    save_processed_data(df)
    X, y = get_clean_data(df, target)

    print("Training model...")
    model, study = train_and_optimize(X, y, target_name=target, n_trials=20)

    print("Saving model and study...")
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(study, "models/optuna_study.pkl")

    print("All done!")

if __name__ == "__main__":
    main()
