"""
Synthetic data generator for OptimalAthlete sprint performance system.
Generates realistic training session data for 400m sprinters.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from setup_db import Athlete, TrainingSession, PerformanceMetric, RaceResult


def generate_synthetic_data(db: Session, num_athletes: int = 5, days_of_data: int = 180):
 """
 Generate synthetic training data for sprint athletes.
 
 Args:
 db: SQLAlchemy database session
 num_athletes: Number of athletes to generate (default 5)
 days_of_data: Number of days of historical data (default 180)
 """
 print(f" Generating synthetic data for {num_athletes} athletes over {days_of_data} days...")
 
 # Clear existing data
 db.query(RaceResult).delete()
 db.query(PerformanceMetric).delete()
 db.query(TrainingSession).delete()
 db.query(Athlete).delete()
 db.commit()
 
 # Generate athletes
 athletes = []
 athlete_names = [
 ("John", "Smith", 46.9),
 ("Sarah", "Johnson", 51.2),
 ("Marcus", "Williams", 45.8),
 ("Emma", "Davis", 52.1),
 ("James", "Wilson", 47.3)
 ]
 
 for i in range(num_athletes):
 first_name, last_name, pb = athlete_names[i]
 athlete = Athlete(
 first_name=first_name,
 last_name=last_name,
 date_of_birth=datetime(2000 + random.randint(0, 4), random.randint(1, 12), random.randint(1, 28)),
 gender="Male" if i % 2 == 0 else "Female",
 personal_best_400m=pb,
 weight_kg=65 + random.randint(-10, 10),
 height_cm=170 + random.randint(-10, 10)
 )
 db.add(athlete)
 athletes.append(athlete)
 
 db.commit()
 print(f" Created {num_athletes} athletes")
 
 # Generate training sessions for each athlete
 start_date = datetime.now() - timedelta(days=days_of_data)
 training_types = ["Speed Endurance", "Tempo", "Speed", "Strength", "Recovery", "Race Pace"]
 
 session_count = 0
 metric_count = 0
 race_count = 0
 
 for athlete in athletes:
 current_date = start_date
 
 # Athlete-specific baseline values
 base_hrv = random.randint(50, 80)
 base_rhr = random.randint(45, 60)
 base_sleep = random.uniform(7.0, 8.5)
 
 while current_date <= datetime.now():
 # Training happens 5-6 days per week
 if random.random() < 0.8: # 80% chance of training on any day
 
 training_type = random.choice(training_types)
 
 # Session intensity varies by type
 intensity_map = {
 "Speed": random.uniform(8.5, 10.0),
 "Speed Endurance": random.uniform(8.0, 9.5),
 "Race Pace": random.uniform(8.5, 9.5),
 "Tempo": random.uniform(6.5, 8.0),
 "Strength": random.uniform(7.0, 8.5),
 "Recovery": random.uniform(3.0, 5.0)
 }
 
 session = TrainingSession(
 athlete_id=athlete.id,
 date=current_date.date(),
 session_type=training_type,
 duration_minutes=random.randint(60, 120),
 intensity_rpe=intensity_map[training_type],
 notes=f"{training_type} session"
 )
 db.add(session)
 db.flush() # Get session.id
 
 session_count += 1
 
 # Add performance metrics (collected most days)
 if random.random() < 0.9: # 90% of training days have metrics
 # Add realistic variation and trends
 fatigue_factor = random.uniform(0.9, 1.1)
 
 metric = PerformanceMetric(
 session_id=session.id,
 hrv_score=int(base_hrv * fatigue_factor),
 resting_heart_rate=int(base_rhr / fatigue_factor),
 sleep_hours=base_sleep * random.uniform(0.85, 1.15),
 sleep_quality=random.randint(6, 10),
 soreness_level=random.randint(1, 7),
 fatigue_level=random.randint(1, 8),
 wellness_score=random.randint(6, 10)
 )
 db.add(metric)
 metric_count += 1
 
 # Add race results (approximately every 2-3 weeks during season)
 if training_type == "Race Pace" and random.random() < 0.15:
 # Performance varies around PB
 time_result = athlete.personal_best_400m + random.uniform(-0.3, 1.5)
 
 race = RaceResult(
 athlete_id=athlete.id,
 date=current_date.date(),
 event="400m",
 time_seconds=time_result,
 position=random.randint(1, 8),
 location=random.choice(["Dublin", "Cork", "Santry", "Belfast", "Galway"]),
 conditions=random.choice(["Good", "Windy", "Rainy", "Perfect", "Hot"])
 )
 db.add(race)
 race_count += 1
 
 current_date += timedelta(days=1)
 
 db.commit()
 
 print(f" Created {session_count} training sessions")
 print(f" Created {metric_count} performance metrics")
 print(f" Created {race_count} race results")
 print(f" Synthetic data generation complete!")


if __name__ == "__main__":
 from database import get_db, init_database
 
 # Initialize database
 init_database()
 
 # Generate data
 db = get_db()
 try:
 generate_synthetic_data(db, num_athletes=5, days_of_data=180)
 finally:
 db.close()