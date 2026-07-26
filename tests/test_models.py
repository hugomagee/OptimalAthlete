"""Tests for model training interface and prediction sanity."""

import json

import numpy as np
import pandas as pd
import pytest

from models import (
    save_models,
    train_random_forest,
    train_xgboost,
)

METRIC_KEYS = {
    'model_name', 'train_mae', 'test_mae',
    'train_rmse', 'test_rmse', 'train_r2', 'test_r2',
}


@pytest.fixture
def regression_data():
    """Small deterministic regression problem shaped like the real dataset."""
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame(
        rng.uniform(0, 10, size=(n, 12)),
        columns=[f'f{i}' for i in range(12)],
    )
    # Race time driven by two features plus noise, centred near 48s
    y = pd.Series(48.0 + 0.3 * X['f0'] - 0.2 * X['f1'] + rng.normal(0, 0.2, n))
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


class TestTrainRandomForest:
    def test_returns_model_and_complete_metrics(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
        assert METRIC_KEYS <= set(metrics.keys())
        assert all(np.isfinite(v) for k, v in metrics.items() if k != 'model_name')
        assert metrics['test_mae'] > 0

    def test_predictions_match_input_shape_and_are_plausible(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        model, _ = train_random_forest(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert np.isfinite(preds).all()
        # Trained on ~48s race times, predictions should stay in that ballpark
        assert ((preds > 40) & (preds < 60)).all()


class TestTrainXgboost:
    def test_returns_model_and_complete_metrics(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
        assert METRIC_KEYS <= set(metrics.keys())
        assert metrics['test_mae'] > 0

    def test_predictions_match_input_shape(self, regression_data):
        X_train, y_train, X_test, y_test = regression_data
        model, _ = train_xgboost(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert np.isfinite(preds).all()


class TestSaveModels:
    def test_writes_model_files_and_metrics(
        self, regression_data, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        X_train, y_train, X_test, y_test = regression_data
        rf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        xg, xg_metrics = train_xgboost(X_train, y_train, X_test, y_test)

        save_models(
            rf, xg, list(X_train.columns),
            metrics={'random_forest': rf_metrics, 'xgboost': xg_metrics},
        )

        assert (tmp_path / 'models' / 'random_forest_model.pkl').exists()
        assert (tmp_path / 'models' / 'xgboost_model.pkl').exists()
        assert (tmp_path / 'models' / 'feature_names.pkl').exists()

        with open(tmp_path / 'models' / 'model_metrics.json') as f:
            saved = json.load(f)
        assert saved['random_forest']['test_mae'] == rf_metrics['test_mae']
