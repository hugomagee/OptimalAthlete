"""
Machine learning models for sprint performance prediction.
Trains Random Forest and XGBoost models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
from feature_engineering import engineer_features


def prepare_ml_data(ml_dataset):
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


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest Regressor.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        Trained model and metrics
    """
    print("\n🌲 Training Random Forest model...")
    
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
    
    print(f"✅ Random Forest trained!")
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


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost Regressor.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        Trained model and metrics
    """
    print("\n🚀 Training XGBoost model...")
    
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
    
    print(f"✅ XGBoost trained!")
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


def display_feature_importance(model, feature_names, model_name):
    """
    Display feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
    """
    print(f"\n📊 {model_name} Feature Importance:")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance_df.to_string(index=False))


def save_models(rf_model, xgb_model, feature_names):
    """
    Save trained models to disk.
    
    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        feature_names: List of feature names
    """
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
    
    print("\n💾 Models saved to 'models/' directory:")
    print("   - random_forest_model.pkl")
    print("   - xgboost_model.pkl")
    print("   - feature_names.pkl")


def train_models():
    """
    Main function to train all ML models.
    """
    print("🤖 Starting ML model training pipeline...\n")
    
    # Load and engineer features
    print("📊 Loading and engineering features...")
    ml_dataset = engineer_features()
    
    # Prepare data
    print("\n🔧 Preparing ML data...")
    X, y, feature_names = prepare_ml_data(ml_dataset)
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    display_feature_importance(rf_model, feature_names, "Random Forest")
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    display_feature_importance(xgb_model, feature_names, "XGBoost")
    
    # Save models
    save_models(rf_model, xgb_model, feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nRandom Forest Test MAE: {rf_metrics['test_mae']:.3f} seconds")
    print(f"XGBoost Test MAE: {xgb_metrics['test_mae']:.3f} seconds")
    
    if xgb_metrics['test_mae'] < rf_metrics['test_mae']:
        print(f"\n🏆 Best model: XGBoost (MAE: {xgb_metrics['test_mae']:.3f}s)")
    else:
        print(f"\n🏆 Best model: Random Forest (MAE: {rf_metrics['test_mae']:.3f}s)")
    
    return rf_model, xgb_model, rf_metrics, xgb_metrics


if __name__ == "__main__":
    train_models()