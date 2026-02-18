"""
Database schema for OptimalAthlete sprint performance system.
Defines tables for athletes, training sessions, performance metrics, and race results.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Athlete(Base):
 """Athlete profile and basic information"""
 __tablename__ = 'athletes'
 
 id = Column(Integer, primary_key=True)
 first_name = Column(String(50), nullable=False)
 last_name = Column(String(50), nullable=False)
 date_of_birth = Column(Date, nullable=False)
 gender = Column(String(10))
 personal_best_400m = Column(Float) # Personal best time in seconds
 weight_kg = Column(Float)
 height_cm = Column(Float)
 created_at = Column(Date, default=datetime.now)


class TrainingSession(Base):
 """Individual training session data"""
 __tablename__ = 'training_sessions'
 
 id = Column(Integer, primary_key=True)
 athlete_id = Column(Integer, nullable=False)
 date = Column(Date, nullable=False)
 session_type = Column(String(50)) # e.g., 'Speed', 'Tempo', 'Strength'
 duration_minutes = Column(Integer)
 intensity_rpe = Column(Float) # Rate of Perceived Exertion (1-10)
 notes = Column(Text)


class PerformanceMetric(Base):
 """Daily performance and recovery metrics"""
 __tablename__ = 'performance_metrics'
 
 id = Column(Integer, primary_key=True)
 session_id = Column(Integer, nullable=False)
 hrv_score = Column(Integer) # Heart Rate Variability
 resting_heart_rate = Column(Integer)
 sleep_hours = Column(Float)
 sleep_quality = Column(Integer) # 1-10 scale
 soreness_level = Column(Integer) # 1-10 scale
 fatigue_level = Column(Integer) # 1-10 scale
 wellness_score = Column(Integer) # 1-10 scale


class RaceResult(Base):
 """Competition race results"""
 __tablename__ = 'race_results'
 
 id = Column(Integer, primary_key=True)
 athlete_id = Column(Integer, nullable=False)
 date = Column(Date, nullable=False)
 event = Column(String(20)) # e.g., '400m'
 time_seconds = Column(Float)
 position = Column(Integer)
 location = Column(String(100))
 conditions = Column(String(50)) # e.g., 'Windy', 'Perfect'


if __name__ == '__main__':
 # Ensure data directory exists
 os.makedirs('data', exist_ok=True)
 
 # Create database
 engine = create_engine('sqlite:///data/optimalathlete.db')
 Base.metadata.create_all(engine)
 
 print("Database initialized at: data/optimalathlete.db")
 print("All tables created successfully!")
 print("- athletes")
 print("- training_sessions")
 print("- performance_metrics")
 print("- race_results")
