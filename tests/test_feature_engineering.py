"""Tests for feature engineering: rolling features and race dataset assembly."""

import numpy as np
import pandas as pd

from feature_engineering import create_training_features, create_race_dataset
from models import prepare_ml_data


EXPECTED_FEATURES = [
    'avg_intensity_7d', 'avg_intensity_14d', 'avg_duration_7d',
    'avg_hrv_7d', 'avg_sleep_7d', 'avg_fatigue_7d',
    'avg_wellness_14d', 'cumulative_load_7d', 'cumulative_load_14d',
    'sessions_past_7d', 'sessions_past_14d', 'avg_recovery_7d',
]


class TestCreateTrainingFeatures:
    def test_adds_all_expected_feature_columns(
        self, sample_sessions_df, sample_metrics_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        for col in EXPECTED_FEATURES:
            assert col in features.columns, f'missing feature column: {col}'

    def test_preserves_all_sessions(self, sample_sessions_df, sample_metrics_df):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        assert len(features) == len(sample_sessions_df)

    def test_rolling_intensity_stays_within_input_bounds(
        self, sample_sessions_df, sample_metrics_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        lo = sample_sessions_df['intensity_rpe'].min()
        hi = sample_sessions_df['intensity_rpe'].max()
        assert features['avg_intensity_7d'].between(lo, hi).all()

    def test_rolling_features_have_no_nans_when_metrics_complete(
        self, sample_sessions_df, sample_metrics_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        assert not features[EXPECTED_FEATURES].isna().any().any()

    def test_cumulative_load_is_intensity_times_duration_summed(
        self, sample_sessions_df, sample_metrics_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        first = features.sort_values(['athlete_id', 'date']).iloc[0]
        expected = first['intensity_rpe'] * first['duration_minutes']
        assert first['cumulative_load_7d'] == expected


class TestCreateRaceDataset:
    def test_output_has_target_and_all_features(
        self, sample_sessions_df, sample_metrics_df, sample_races_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        ml_df = create_race_dataset(sample_races_df, features)
        assert 'race_time_seconds' in ml_df.columns
        for col in EXPECTED_FEATURES:
            assert col in ml_df.columns

    def test_uses_only_data_from_before_the_race(
        self, sample_sessions_df, sample_metrics_df, sample_races_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        ml_df = create_race_dataset(sample_races_df, features)
        # Athlete 1 raced on Jan 15; the matched features must come from the
        # most recent session strictly before that date (Jan 14 → 14 sessions).
        jan_15 = ml_df[ml_df['race_date'] == pd.Timestamp('2025-01-15')].iloc[0]
        assert jan_15['sessions_past_7d'] <= 7
        assert jan_15['race_time_seconds'] == 47.5

    def test_drops_races_with_no_prior_training_data(
        self, sample_sessions_df, sample_metrics_df, sample_races_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        ml_df = create_race_dataset(sample_races_df, features)
        # Athlete 3 has no training sessions, so their race must be dropped.
        assert 3 not in ml_df['athlete_id'].values
        assert len(ml_df) == 2


class TestPrepareMlData:
    def test_returns_aligned_features_and_target(
        self, sample_sessions_df, sample_metrics_df, sample_races_df
    ):
        features = create_training_features(sample_sessions_df, sample_metrics_df)
        ml_df = create_race_dataset(sample_races_df, features)
        X, y, feature_names = prepare_ml_data(ml_df)
        assert list(X.columns) == feature_names
        assert len(feature_names) == 12
        assert len(X) == len(y)
        assert np.isfinite(X.to_numpy(dtype=float)).all()
