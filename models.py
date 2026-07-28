"""
Machine learning models for sprint performance prediction.

Two evaluation protocols are computed and both are persisted, because the
difference between them is the point:

- **Walk-forward** (the reported protocol). Races are sorted by date; each race
  is predicted by a model trained only on races that happened before it. This
  is the only protocol that answers the question the system actually poses —
  "given training data up to today, what will the next race time be?"

- **Naive pooled split** (a cautionary comparison, not a result). A random
  train/test split over pooled athletes, which is what this project originally
  reported. It leaks in two directions at once: temporally, because future
  races inform predictions of past ones, and by athlete identity, because the
  same athlete appears on both sides of the split and most of the variance in
  pooled race times is *between* athletes rather than within them. The analysis
  notebook quantifies this; the number it produces is inflated and is labelled
  as such everywhere it appears.

A "predict the athlete's recent average" baseline is evaluated under the same
walk-forward protocol, because a model that cannot beat that baseline has not
demonstrated anything.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_engineering import engineer_features


def prepare_ml_data(ml_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare features and target for ML training.

    Args:
        ml_dataset: DataFrame with engineered features

    Returns:
        tuple: (X, y, feature_names)
    """
    # Define feature columns (exclude ID, date, and target)
    feature_cols = [
        'avg_intensity_7d', 'avg_intensity_14d', 'avg_duration_7d',
        'avg_hrv_7d', 'avg_sleep_7d', 'avg_fatigue_7d',
        'avg_wellness_14d', 'cumulative_load_7d', 'cumulative_load_14d',
        'sessions_past_7d', 'sessions_past_14d', 'avg_recovery_7d'
    ]

    X = ml_dataset[feature_cols]
    y = ml_dataset['race_time_seconds']

    return X, y, feature_cols


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[RandomForestRegressor, dict]:
    """
    Train Random Forest Regressor.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained model and metrics
    """
    print("\nTraining Random Forest model...")

    # Initialize model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    # Train model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("Random Forest trained!")
    print(f"   Training MAE: {train_mae:.3f} seconds")
    print(f"   Test MAE: {test_mae:.3f} seconds")
    print(f"   Test RMSE: {test_rmse:.3f} seconds")
    print(f"   Test R²: {test_r2:.3f}")

    metrics = {
        'model_name': 'Random Forest',
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

    return rf_model, metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost Regressor.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained model and metrics
    """
    print("\nTraining XGBoost model...")

    # Initialize model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    # Train model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("XGBoost trained!")
    print(f"   Training MAE: {train_mae:.3f} seconds")
    print(f"   Test MAE: {test_mae:.3f} seconds")
    print(f"   Test RMSE: {test_rmse:.3f} seconds")
    print(f"   Test R²: {test_r2:.3f}")

    metrics = {
        'model_name': 'XGBoost',
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

    return xgb_model, metrics


def _score(y_true, y_pred, label: str, n_train_note: str = '') -> dict:
    """Standard metric bundle for a set of out-of-sample predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        'model_name': label,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'n_predictions': int(len(y_true)),
    }
    # R^2 is undefined for a single point and unstable for very few; report it
    # only where it can carry meaning, and let it go negative when it should.
    out['r2'] = float(r2_score(y_true, y_pred)) if len(y_true) >= 3 else None
    if n_train_note:
        out['note'] = n_train_note
    return out


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=2,
        min_samples_leaf=1, random_state=42,
    )


def make_xgboost() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
    )


def walk_forward_predict(
    ml_dataset: pd.DataFrame,
    feature_cols: list[str],
    model_factory,
    min_train: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expanding-window walk-forward prediction over races in date order.

    For each race i (from ``min_train`` onward) a fresh model is fitted on races
    0..i-1 and used to predict race i. No future information reaches any
    prediction.

    Returns:
        (y_true, y_pred) for the predicted races. Empty arrays if the dataset is
        too small to leave any out-of-sample races.
    """
    ordered = ml_dataset.sort_values('race_date').reset_index(drop=True)
    X = ordered[feature_cols]
    y = ordered['race_time_seconds']

    if len(ordered) <= min_train:
        return np.array([]), np.array([])

    y_true, y_pred = [], []
    for i in range(min_train, len(ordered)):
        model = model_factory()
        model.fit(X.iloc[:i], y.iloc[:i])
        y_pred.append(float(model.predict(X.iloc[[i]])[0]))
        y_true.append(float(y.iloc[i]))

    return np.array(y_true), np.array(y_pred)


def walk_forward_baseline(
    ml_dataset: pd.DataFrame,
    min_train: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    "Predict the athlete's recent average" baseline, same walk-forward protocol.

    For each race, predict the mean of that athlete's previous race times,
    falling back to the mean over all previous races when the athlete has none.
    A model that cannot beat this has not learned anything about form.
    """
    ordered = ml_dataset.sort_values('race_date').reset_index(drop=True)
    if len(ordered) <= min_train:
        return np.array([]), np.array([])

    y_true, y_pred = [], []
    for i in range(min_train, len(ordered)):
        history = ordered.iloc[:i]
        athlete_id = ordered.iloc[i]['athlete_id']
        own = history[history['athlete_id'] == athlete_id]['race_time_seconds']
        prediction = own.mean() if len(own) else history['race_time_seconds'].mean()
        y_pred.append(float(prediction))
        y_true.append(float(ordered.iloc[i]['race_time_seconds']))

    return np.array(y_true), np.array(y_pred)


def evaluate_walk_forward(ml_dataset: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Run every model plus the baseline under the walk-forward protocol."""
    results = {}

    for key, label, factory in [
        ('random_forest', 'Random Forest', make_random_forest),
        ('xgboost', 'XGBoost', make_xgboost),
    ]:
        y_true, y_pred = walk_forward_predict(ml_dataset, feature_cols, factory)
        if len(y_true):
            results[key] = _score(y_true, y_pred, label)

    y_true, y_pred = walk_forward_baseline(ml_dataset)
    if len(y_true):
        results['baseline_recent_average'] = _score(
            y_true, y_pred, 'Baseline: athlete recent average'
        )

    return results


def display_feature_importance(model, feature_names: list[str], model_name: str) -> None:
    """
    Display feature importance from trained model.

    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
    """
    print(f"\n{model_name} Feature Importance:")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(feature_importance_df.to_string(index=False))


def save_models(
    rf_model: RandomForestRegressor,
    xgb_model: xgb.XGBRegressor,
    feature_names: list[str],
    metrics: dict | None = None,
) -> None:
    """
    Save trained models to disk.

    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        feature_names: List of feature names
        metrics: Optional dict of evaluation metrics to save alongside models
    """
    # TODO: Add model versioning and metadata tracking
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save Random Forest
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    # Save XGBoost
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # Save evaluation metrics so the dashboard reports real numbers
    if metrics is not None:
        with open('models/model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    print("\nModels saved to 'models/' directory:")
    print("   - random_forest_model.pkl")
    print("   - xgboost_model.pkl")
    print("   - feature_names.pkl")
    print("   - model_metrics.json")


def train_models() -> tuple[RandomForestRegressor, xgb.XGBRegressor, dict, dict]:
    """
    Main function to train all ML models.
    """
    print("Starting ML model training pipeline...\n")

    # Load and engineer features
    print("Loading and engineering features...")
    ml_dataset = engineer_features()

    # Prepare data
    print("\nPreparing ML data...")
    X, y, feature_names = prepare_ml_data(ml_dataset)
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X)}")

    # ── Reported protocol: walk-forward ────────────────────────────────────
    print("\nEvaluating with walk-forward validation (the reported protocol)...")
    print("   Each race is predicted using only races that happened before it.")
    walk_forward = evaluate_walk_forward(ml_dataset, feature_names)

    if walk_forward:
        for key, m in walk_forward.items():
            r2 = f"{m['r2']:.3f}" if m['r2'] is not None else "n/a"
            print(f"   {m['model_name']:<34} MAE {m['mae']:.3f}s  "
                  f"RMSE {m['rmse']:.3f}s  R² {r2}  (n={m['n_predictions']})")
    else:
        print("   Not enough races for walk-forward evaluation.")

    # ── Cautionary comparison: the leaky pooled split ──────────────────────
    print("\nFor comparison only — naive pooled random split (LEAKY, not a result):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    display_feature_importance(rf_model, feature_names, "Random Forest")

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    display_feature_importance(xgb_model, feature_names, "XGBoost")

    metrics = {
        'reported_protocol': 'walk_forward',
        'walk_forward': walk_forward,
        'naive_pooled_split': {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'warning': (
                'Leaky protocol, retained only as a cautionary comparison. Pools '
                'athletes and splits at random, so the same athlete appears on both '
                'sides of the split and future races inform predictions of past ones. '
                'Do not report these numbers as model performance.'
            ),
        },
        'n_samples': int(len(X)),
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }

    save_models(rf_model, xgb_model, feature_names, metrics=metrics)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

    if walk_forward:
        best_key = min(
            (k for k in walk_forward if k != 'baseline_recent_average'),
            key=lambda k: walk_forward[k]['mae'],
            default=None,
        )
        baseline = walk_forward.get('baseline_recent_average')
        if best_key:
            best = walk_forward[best_key]
            print(f"\nBest model under walk-forward: {best['model_name']} "
                  f"(MAE {best['mae']:.3f}s over {best['n_predictions']} races)")
            if baseline:
                print(f"Recent-average baseline:       MAE {baseline['mae']:.3f}s")
                if best['mae'] < baseline['mae']:
                    print("The model beats the baseline on this data.")
                else:
                    print("The model does NOT beat the baseline on this data — "
                          "it has not demonstrated predictive value.")

    print(f"\nNaive pooled split, for contrast: "
          f"RF test MAE {rf_metrics['test_mae']:.3f}s, "
          f"R² {rf_metrics['test_r2']:.3f} (leaky — not a result)")

    return rf_model, xgb_model, rf_metrics, xgb_metrics


if __name__ == "__main__":
    train_models()
