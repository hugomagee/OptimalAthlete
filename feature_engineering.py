"""
Feature engineering for sprint performance prediction.
Transforms raw training data into ML-ready features.
"""


import pandas as pd

from database import get_db
from setup_db import Athlete, PerformanceMetric, RaceResult, TrainingSession


def load_data_from_db() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data from database into pandas DataFrames.

    Returns:
        tuple: (athletes_df, sessions_df, metrics_df, races_df)
    """
    db = get_db()

    try:
        # Load athletes
        athletes = db.query(Athlete).all()
        athletes_df = pd.DataFrame([{
            'athlete_id': a.id,
            'first_name': a.first_name,
            'last_name': a.last_name,
            'gender': a.gender,
            'personal_best_400m': a.personal_best_400m,
            'weight_kg': a.weight_kg,
            'height_cm': a.height_cm
        } for a in athletes])

        # Load training sessions
        sessions = db.query(TrainingSession).all()
        sessions_df = pd.DataFrame([{
            'session_id': s.id,
            'athlete_id': s.athlete_id,
            'date': s.date,
            'session_type': s.session_type,
            'duration_minutes': s.duration_minutes,
            'intensity_rpe': s.intensity_rpe
        } for s in sessions])

        # Load performance metrics
        metrics = db.query(PerformanceMetric).all()
        metrics_df = pd.DataFrame([{
            'session_id': m.session_id,
            'hrv_score': m.hrv_score,
            'resting_heart_rate': m.resting_heart_rate,
            'sleep_hours': m.sleep_hours,
            'sleep_quality': m.sleep_quality,
            'soreness_level': m.soreness_level,
            'fatigue_level': m.fatigue_level,
            'wellness_score': m.wellness_score
        } for m in metrics])

        # Load race results
        races = db.query(RaceResult).all()
        races_df = pd.DataFrame([{
            'race_id': r.id,
            'athlete_id': r.athlete_id,
            'date': r.date,
            'event': r.event,
            'time_seconds': r.time_seconds,
            'position': r.position,
            'location': r.location
        } for r in races])

        return athletes_df, sessions_df, metrics_df, races_df

    finally:
        db.close()


def create_training_features(
    sessions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    short_window: str = '7D',
    long_window: str = '14D',
) -> pd.DataFrame:
    """
    Create rolling features from training sessions using genuine time windows.

    The windows are calendar-based ('7D' = the trailing seven days, inclusive of
    the current session), not row-based. This distinction matters: an athlete
    who trains twice a week and one who trains daily have very different
    seven-day loads, but a seven-ROW window would treat them identically while
    still being labelled "7d".

    A bug fixed in this function: these features previously used
    ``rolling(window=7)``, which counts sessions rather than days, so
    ``sessions_past_7d`` was structurally incapable of varying (it counted rows
    in a 7-row window, i.e. min(7, sessions so far)) and every ``*_7d`` name was
    inaccurate. See the changelog note in the README.

    Args:
        sessions_df: DataFrame of training sessions
        metrics_df: DataFrame of performance metrics
        short_window: pandas offset alias for the short lookback (default '7D')
        long_window: pandas offset alias for the long lookback (default '14D')

    Returns:
        DataFrame with engineered features, one row per session
    """
    # Merge sessions with metrics
    df = sessions_df.merge(metrics_df, on='session_id', how='left')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['athlete_id', 'date'])

    # Derived per-session quantities the rolling windows aggregate
    df['training_load'] = df['intensity_rpe'] * df['duration_minutes']
    df['recovery_score'] = (10 - df['fatigue_level']) * df['sleep_hours'] / 8

    features_list = []

    for _, athlete_df in df.groupby('athlete_id', sort=False):
        # Time-based rolling requires a sorted DatetimeIndex.
        athlete_df = athlete_df.sort_values('date').set_index('date')
        short = athlete_df.rolling(short_window)
        long = athlete_df.rolling(long_window)

        # Trailing 7-day averages
        athlete_df['avg_intensity_7d'] = short['intensity_rpe'].mean()
        athlete_df['avg_duration_7d'] = short['duration_minutes'].mean()
        athlete_df['avg_hrv_7d'] = short['hrv_score'].mean()
        athlete_df['avg_sleep_7d'] = short['sleep_hours'].mean()
        athlete_df['avg_fatigue_7d'] = short['fatigue_level'].mean()
        athlete_df['avg_recovery_7d'] = short['recovery_score'].mean()

        # Trailing 14-day averages
        athlete_df['avg_intensity_14d'] = long['intensity_rpe'].mean()
        athlete_df['avg_wellness_14d'] = long['wellness_score'].mean()

        # Accumulated training load
        athlete_df['cumulative_load_7d'] = short['training_load'].sum()
        athlete_df['cumulative_load_14d'] = long['training_load'].sum()

        # Session counts — now genuinely "how many sessions in the last N days"
        athlete_df['sessions_past_7d'] = short['session_id'].count()
        athlete_df['sessions_past_14d'] = long['session_id'].count()

        features_list.append(athlete_df.reset_index())

    # Combine all athletes
    features_df = pd.concat(features_list, ignore_index=True)

    return features_df


def create_race_dataset(races_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML dataset by matching race dates with training features.

    Args:
        races_df: DataFrame of race results
        features_df: DataFrame with engineered training features

    Returns:
        DataFrame ready for ML modeling
    """
    ml_dataset = []

    for _, race in races_df.iterrows():
        athlete_id = race['athlete_id']
        race_date = pd.to_datetime(race['date'])

        # Get features from 1 day before race (most recent training data)
        athlete_features = features_df[
            (features_df['athlete_id'] == athlete_id) &
            (features_df['date'] < race_date)
        ].sort_values('date').tail(1)

        if len(athlete_features) > 0:
            features = athlete_features.iloc[0]

            ml_dataset.append({
                'athlete_id': athlete_id,
                'race_date': race_date,
                'race_time_seconds': race['time_seconds'],
                'avg_intensity_7d': features['avg_intensity_7d'],
                'avg_intensity_14d': features['avg_intensity_14d'],
                'avg_duration_7d': features['avg_duration_7d'],
                'avg_hrv_7d': features['avg_hrv_7d'],
                'avg_sleep_7d': features['avg_sleep_7d'],
                'avg_fatigue_7d': features['avg_fatigue_7d'],
                'avg_wellness_14d': features['avg_wellness_14d'],
                'cumulative_load_7d': features['cumulative_load_7d'],
                'cumulative_load_14d': features['cumulative_load_14d'],
                'sessions_past_7d': features['sessions_past_7d'],
                'sessions_past_14d': features['sessions_past_14d'],
                'avg_recovery_7d': features['avg_recovery_7d']
            })

    ml_df = pd.DataFrame(ml_dataset)

    # Remove any rows with missing values
    ml_df = ml_df.dropna()

    return ml_df


def engineer_features() -> pd.DataFrame:
    """
    Main function to run complete feature engineering pipeline.

    Returns:
        DataFrame ready for ML modeling
    """
    print("Starting feature engineering...")

    # Load data
    print("Loading data from database...")
    athletes_df, sessions_df, metrics_df, races_df = load_data_from_db()

    print(f"   - {len(athletes_df)} athletes")
    print(f"   - {len(sessions_df)} training sessions")
    print(f"   - {len(metrics_df)} performance metrics")
    print(f"   - {len(races_df)} race results")

    # Create training features
    print("Creating training features...")
    features_df = create_training_features(sessions_df, metrics_df)

    # Create ML dataset
    print("Creating ML dataset...")
    ml_dataset = create_race_dataset(races_df, features_df)

    print("Feature engineering complete!")
    print(f"   - {len(ml_dataset)} race samples with features")
    print(f"   - {len(ml_dataset.columns) - 3} features per sample")

    return ml_dataset


if __name__ == "__main__":
    # Run feature engineering and display results
    ml_dataset = engineer_features()

    print("\nSample of engineered features:")
    print(ml_dataset.head())

    print("\nDataset info:")
    print(f"Shape: {ml_dataset.shape}")
    print(f"Columns: {list(ml_dataset.columns)}")
