"""Tests for the walk-forward evaluation protocol and the rolling-window fix.

These cover the two corrections that changed what this project reports:

1. Features use genuine calendar windows, not row counts.
2. The headline protocol is walk-forward, and it is verifiably free of the
   temporal leakage that the pooled random split introduces.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import create_training_features
from models import (
    evaluate_walk_forward,
    make_random_forest,
    walk_forward_baseline,
    walk_forward_predict,
)

FEATURES = [
    'avg_intensity_7d', 'avg_intensity_14d', 'avg_duration_7d',
    'avg_hrv_7d', 'avg_sleep_7d', 'avg_fatigue_7d',
    'avg_wellness_14d', 'cumulative_load_7d', 'cumulative_load_14d',
    'sessions_past_7d', 'sessions_past_14d', 'avg_recovery_7d',
]


def _sessions(athlete_specs):
    """Build sessions where each athlete trains every `step` days."""
    rows, sid = [], 0
    for athlete_id, step, n in athlete_specs:
        for k in range(n):
            sid += 1
            rows.append({
                'session_id': sid,
                'athlete_id': athlete_id,
                'date': pd.Timestamp('2025-01-01') + pd.Timedelta(days=k * step),
                'session_type': 'Tempo',
                'duration_minutes': 60,
                'intensity_rpe': 7.0,
            })
    return pd.DataFrame(rows)


def _metrics(sessions_df):
    return pd.DataFrame([{
        'session_id': sid, 'hrv_score': 60, 'resting_heart_rate': 50,
        'sleep_hours': 8.0, 'sleep_quality': 8, 'soreness_level': 3,
        'fatigue_level': 4, 'wellness_score': 8,
    } for sid in sessions_df['session_id']])


# ── rolling windows are calendar-based ────────────────────────────────────

def test_session_counts_reflect_training_frequency_not_row_position():
    """Regression: rolling(window=7) counted ROWS, so an athlete training daily
    and one training every 4 days both reported the same sessions_past_7d."""
    sessions = _sessions([(1, 1, 20), (2, 4, 20)])   # daily vs every 4 days
    features = create_training_features(sessions, _metrics(sessions))

    daily = features[features['athlete_id'] == 1]['sessions_past_7d']
    sparse = features[features['athlete_id'] == 2]['sessions_past_7d']

    # Daily trainer approaches 7 sessions per 7-day window; the sparse one ~2.
    assert daily.max() == 7
    assert sparse.max() <= 2
    assert daily.mean() > sparse.mean() * 2


def test_seven_day_window_covers_exactly_seven_calendar_days():
    sessions = _sessions([(1, 1, 15)])
    features = create_training_features(sessions, _metrics(sessions))
    # By the 15th daily session the trailing 7-day window holds 7 sessions
    # (today plus the previous six days).
    assert features['sessions_past_7d'].iloc[-1] == 7
    assert features['sessions_past_14d'].iloc[-1] == 14


def test_cumulative_load_scales_with_sessions_in_window():
    sessions = _sessions([(1, 1, 15)])
    features = create_training_features(sessions, _metrics(sessions))
    last = features.iloc[-1]
    # 7 sessions x (7.0 RPE x 60 min) = 2940
    assert last['cumulative_load_7d'] == pytest.approx(7 * 7.0 * 60)


def test_all_expected_features_present_after_the_window_fix():
    sessions = _sessions([(1, 1, 10)])
    features = create_training_features(sessions, _metrics(sessions))
    for col in FEATURES:
        assert col in features.columns


# ── walk-forward protocol ─────────────────────────────────────────────────

@pytest.fixture
def race_dataset():
    """20 races in date order with features and a per-athlete mean race time."""
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame({
        'athlete_id': [1 if i % 2 == 0 else 2 for i in range(n)],
        'race_date': pd.date_range('2025-01-01', periods=n, freq='7D'),
        'race_time_seconds': [
            (47.0 if i % 2 == 0 else 51.0) + rng.normal(0, 0.3) for i in range(n)
        ],
        **{f: rng.uniform(1, 10, n) for f in FEATURES},
    })


def test_walk_forward_predicts_every_race_after_the_minimum_training_window(race_dataset):
    y_true, y_pred = walk_forward_predict(
        race_dataset, FEATURES, make_random_forest, min_train=10
    )
    assert len(y_true) == len(race_dataset) - 10
    assert len(y_pred) == len(y_true)
    assert np.isfinite(y_pred).all()


def test_walk_forward_uses_only_past_data():
    """A model trained only on the past cannot see a future level shift.

    Race times jump by 10 seconds exactly halfway through. A leaky protocol
    would partly fit the jump; walk-forward must under-predict the first races
    after it, because nothing before the jump contains that information.
    """
    n = 24
    dataset = pd.DataFrame({
        'athlete_id': 1,
        'race_date': pd.date_range('2025-01-01', periods=n, freq='7D'),
        'race_time_seconds': [45.0] * (n // 2) + [55.0] * (n // 2),
        **{f: np.linspace(1, 10, n) for f in FEATURES},
    })
    y_true, y_pred = walk_forward_predict(
        dataset, FEATURES, make_random_forest, min_train=10
    )
    first_after_jump = np.argmax(y_true > 50)
    # The prediction for the first post-jump race is trained purely on 45s races.
    assert y_pred[first_after_jump] < 50


def test_walk_forward_returns_empty_when_dataset_is_too_small(race_dataset):
    y_true, y_pred = walk_forward_predict(
        race_dataset.head(5), FEATURES, make_random_forest, min_train=10
    )
    assert len(y_true) == 0
    assert len(y_pred) == 0


def test_baseline_predicts_each_athletes_own_history(race_dataset):
    y_true, y_pred = walk_forward_baseline(race_dataset, min_train=10)
    assert len(y_true) == len(race_dataset) - 10
    # Athlete 1 races ~47s, athlete 2 ~51s; the baseline must separate them.
    athletes = race_dataset.sort_values('race_date')['athlete_id'].to_numpy()[10:]
    assert y_pred[athletes == 1].mean() < y_pred[athletes == 2].mean()


def test_evaluate_walk_forward_reports_models_and_baseline(race_dataset):
    results = evaluate_walk_forward(race_dataset, FEATURES)
    assert {'random_forest', 'xgboost', 'baseline_recent_average'} <= set(results)
    for entry in results.values():
        assert entry['mae'] > 0
        assert entry['n_predictions'] == len(race_dataset) - 10


def test_walk_forward_metrics_carry_prediction_counts(race_dataset):
    """Small out-of-sample counts must be visible, not hidden behind a ratio."""
    results = evaluate_walk_forward(race_dataset, FEATURES)
    for entry in results.values():
        assert isinstance(entry['n_predictions'], int)
        assert 'r2' in entry
