"""Tests for synthetic data generation and database loading."""

import pandas as pd

from data_loader import generate_synthetic_data
from setup_db import Athlete, TrainingSession, PerformanceMetric, RaceResult


class TestGenerateSyntheticData:
    def test_creates_requested_number_of_athletes(self, db_session):
        generate_synthetic_data(db_session, num_athletes=3, days_of_data=30)
        assert db_session.query(Athlete).count() == 3

    def test_populates_all_tables(self, populated_db):
        assert populated_db.query(TrainingSession).count() > 0
        assert populated_db.query(PerformanceMetric).count() > 0
        # Races are probabilistic but 3 athletes x 60 days virtually always
        # produce at least one; athletes/sessions are the hard guarantees.
        assert populated_db.query(Athlete).count() == 3

    def test_is_idempotent_replaces_existing_data(self, db_session):
        generate_synthetic_data(db_session, num_athletes=2, days_of_data=20)
        generate_synthetic_data(db_session, num_athletes=3, days_of_data=20)
        assert db_session.query(Athlete).count() == 3

    def test_session_values_are_plausible(self, populated_db):
        for s in populated_db.query(TrainingSession).all():
            assert 0 < s.duration_minutes <= 180
            assert 0 < s.intensity_rpe <= 10
            assert s.athlete_id is not None

    def test_race_times_are_plausible_400m_times(self, db_session):
        generate_synthetic_data(db_session, num_athletes=5, days_of_data=120)
        races = db_session.query(RaceResult).all()
        for r in races:
            assert 40 < r.time_seconds < 60
            assert r.event == '400m'

    def test_metrics_reference_existing_sessions(self, populated_db):
        session_ids = {s.id for s in populated_db.query(TrainingSession).all()}
        for m in populated_db.query(PerformanceMetric).all():
            assert m.session_id in session_ids


class TestLoadDataFromDb:
    def test_returns_four_dataframes_with_expected_columns(
        self, populated_db, monkeypatch
    ):
        import feature_engineering
        monkeypatch.setattr(feature_engineering, 'get_db', lambda: populated_db)

        athletes, sessions, metrics, races = feature_engineering.load_data_from_db()

        assert isinstance(athletes, pd.DataFrame) and len(athletes) == 3
        assert {'athlete_id', 'personal_best_400m'} <= set(athletes.columns)
        assert {'session_id', 'athlete_id', 'date', 'intensity_rpe'} <= set(
            sessions.columns
        )
        assert {'session_id', 'hrv_score', 'sleep_hours'} <= set(metrics.columns)
        assert {'athlete_id', 'date', 'time_seconds'} <= set(races.columns)
