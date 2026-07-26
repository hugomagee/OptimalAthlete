"""Shared fixtures for the OptimalAthlete test suite."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from setup_db import Base


@pytest.fixture
def db_session():
    """SQLAlchemy session bound to a fresh in-memory SQLite database."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def populated_db(db_session):
    """In-memory database populated with a small synthetic dataset."""
    from data_loader import generate_synthetic_data
    generate_synthetic_data(db_session, num_athletes=3, days_of_data=60)
    return db_session


@pytest.fixture
def sample_sessions_df():
    """Hand-built training sessions for two athletes over consecutive days."""
    rows = []
    session_id = 1
    for athlete_id in (1, 2):
        for day in range(1, 21):
            rows.append({
                'session_id': session_id,
                'athlete_id': athlete_id,
                'date': f'2025-01-{day:02d}',
                'session_type': 'Tempo',
                'duration_minutes': 60 + day,
                'intensity_rpe': 5.0 + (day % 5),
            })
            session_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def sample_metrics_df(sample_sessions_df):
    """Metrics matching every session in sample_sessions_df."""
    return pd.DataFrame([{
        'session_id': sid,
        'hrv_score': 60,
        'resting_heart_rate': 50,
        'sleep_hours': 8.0,
        'sleep_quality': 8,
        'soreness_level': 3,
        'fatigue_level': 4,
        'wellness_score': 8,
    } for sid in sample_sessions_df['session_id']])


@pytest.fixture
def sample_races_df():
    """Races for athlete 1 (mid-series and post-series) and athlete 3 (no sessions)."""
    return pd.DataFrame([
        {'athlete_id': 1, 'date': '2025-01-15', 'time_seconds': 47.5},
        {'athlete_id': 1, 'date': '2025-02-01', 'time_seconds': 48.2},
        {'athlete_id': 3, 'date': '2025-01-10', 'time_seconds': 50.0},
    ])
