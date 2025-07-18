# train_model.py

import numpy as np
import xgboost as xgb
import optuna
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === CONFIGURATION ===
MODEL_PATH = Path("models/xgb_model.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.pkl")
N_SPLITS = 5
RANDOM_STATE = 42


def train_and_optimize(X, y, target_name="DE_load_actual_entsoe_transparency", n_trials=20):
    """Optimize and train an XGBoost model using TimeSeriesSplit and Optuna."""

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial):
        """Objective function for Optuna optimization with time-aware CV."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': RANDOM_STATE,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }

        rmses = []

        for train_idx, valid_idx in tscv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            rmses.append(rmse)

        return np.mean(rmses)

    # === Run Optuna Study ===
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("\n Best Trial:")
    print(study.best_trial)

    # === Train Final Model on All Data ===
    best_params = study.best_trial.params
    best_params.update({
        'random_state': RANDOM_STATE,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    })

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y)

    # === Save Model and Feature List ===
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_NAMES_PATH)

    # === Print Final Metrics on Last Fold (for transparency) ===
    last_train_idx, last_valid_idx = list(tscv.split(X))[-1]
    X_train_last, X_valid_last = X.iloc[last_train_idx], X.iloc[last_valid_idx]
    y_train_last, y_valid_last = y.iloc[last_train_idx], y.iloc[last_valid_idx]
    y_pred = final_model.predict(X_valid_last)

    print("\nFinal Model Performance on Last Validation Fold:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_valid_last, y_pred)):.3f}")
    print(f"MAE: {mean_absolute_error(y_valid_last, y_pred):.3f}")
    print(f"R²: {r2_score(y_valid_last, y_pred):.3f}")

    return final_model, study
