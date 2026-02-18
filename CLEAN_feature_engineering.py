"""
Feature engineering for sprint performance prediction.
Transforms raw training data into ML-ready features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database import get_db
from setup_db import Athlete, TrainingSession, PerformanceMetric, RaceResult


def load_data_from_db():
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


def create_training_features(sessions_df, metrics_df, lookback_days=14):
 """
 Create rolling features from training sessions.
 
 Args:
 sessions_df: DataFrame of training sessions
 metrics_df: DataFrame of performance metrics
 lookback_days: Number of days to look back for rolling features
 
 Returns:
 DataFrame with engineered features
 """
 # Merge sessions with metrics
 df = sessions_df.merge(metrics_df, on='session_id', how='left')
 df['date'] = pd.to_datetime(df['date'])
 df = df.sort_values(['athlete_id', 'date'])
 
 # Create features for each athlete
 features_list = []
 
 for athlete_id in df['athlete_id'].unique():
 athlete_df = df[df['athlete_id'] == athlete_id].copy()
 athlete_df = athlete_df.sort_values('date')
 
 # Rolling averages (past 7 days)
 athlete_df['avg_intensity_7d'] = athlete_df['intensity_rpe'].rolling(window=7, min_periods=1).mean()
 athlete_df['avg_duration_7d'] = athlete_df['duration_minutes'].rolling(window=7, min_periods=1).mean()
 athlete_df['avg_hrv_7d'] = athlete_df['hrv_score'].rolling(window=7, min_periods=1).mean()
 athlete_df['avg_sleep_7d'] = athlete_df['sleep_hours'].rolling(window=7, min_periods=1).mean()
 athlete_df['avg_fatigue_7d'] = athlete_df['fatigue_level'].rolling(window=7, min_periods=1).mean()
 
 # Rolling averages (past 14 days)
 athlete_df['avg_intensity_14d'] = athlete_df['intensity_rpe'].rolling(window=14, min_periods=1).mean()
 athlete_df['avg_wellness_14d'] = athlete_df['wellness_score'].rolling(window=14, min_periods=1).mean()
 
 # Training load (intensity × duration)
 athlete_df['training_load'] = athlete_df['intensity_rpe'] * athlete_df['duration_minutes']
 athlete_df['cumulative_load_7d'] = athlete_df['training_load'].rolling(window=7, min_periods=1).sum()
 athlete_df['cumulative_load_14d'] = athlete_df['training_load'].rolling(window=14, min_periods=1).sum()
 
 # Session count features
 athlete_df['sessions_past_7d'] = athlete_df['session_id'].rolling(window=7, min_periods=1).count()
 athlete_df['sessions_past_14d'] = athlete_df['session_id'].rolling(window=14, min_periods=1).count()
 
 # Recovery score (inverse of fatigue, combined with sleep)
 athlete_df['recovery_score'] = (10 - athlete_df['fatigue_level']) * athlete_df['sleep_hours'] / 8
 athlete_df['avg_recovery_7d'] = athlete_df['recovery_score'].rolling(window=7, min_periods=1).mean()
 
 features_list.append(athlete_df)
 
 # Combine all athletes
 features_df = pd.concat(features_list, ignore_index=True)
 
 return features_df


def create_race_dataset(races_df, features_df):
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


def engineer_features():
 """
 Main function to run complete feature engineering pipeline.
 
 Returns:
 DataFrame ready for ML modeling
 """
 print("Starting feature engineering...")
 
 # Load data
 print("Loading data from database...")
 athletes_df, sessions_df, metrics_df, races_df = load_data_from_db()
 
 print(f" - {len(athletes_df)} athletes")
 print(f" - {len(sessions_df)} training sessions")
 print(f" - {len(metrics_df)} performance metrics")
 print(f" - {len(races_df)} race results")
 
 # Create training features
 print("Creating training features...")
 features_df = create_training_features(sessions_df, metrics_df)
 
 # Create ML dataset
 print("Creating ML dataset...")
 ml_dataset = create_race_dataset(races_df, features_df)
 
 print(f" Feature engineering complete!")
 print(f" - {len(ml_dataset)} race samples with features")
 print(f" - {len(ml_dataset.columns) - 3} features per sample") # -3 for athlete_id, race_date, race_time
 
 return ml_dataset


if __name__ == "__main__":
 # Run feature engineering and display results
 ml_dataset = engineer_features()
 
 print("\n Sample of engineered features:")
 print(ml_dataset.head())
 
 print("\n Dataset info:")
 print(f"Shape: {ml_dataset.shape}")
 print(f"Columns: {list(ml_dataset.columns)}")
