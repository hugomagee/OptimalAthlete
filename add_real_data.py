"""
Add Hugo Magee's real 2025 race data to the database.
Replaces synthetic data with actual race results.
"""

from database import get_db
from setup_db import Athlete, TrainingSession, PerformanceMetric, RaceResult
from datetime import datetime, timedelta
import random


def clear_existing_data():
    """Clear existing data for athlete ID 1."""
    db = get_db()
    try:
        # Delete in correct order (foreign key constraints)
        db.query(RaceResult).filter(RaceResult.athlete_id == 1).delete()
        db.query(PerformanceMetric).filter(PerformanceMetric.session_id.in_(
            db.query(TrainingSession.id).filter(TrainingSession.athlete_id == 1)
        )).delete()
        db.query(TrainingSession).filter(TrainingSession.athlete_id == 1).delete()
        db.commit()
        print("✅ Cleared existing data")
    finally:
        db.close()


def add_real_races():
    """Add real 2025 race results."""
    db = get_db()
    
    # Real race data from World Athletics
    races = [
        # (date, time_seconds, location, conditions)
        ("2025-04-12", 47.68, "Marsa Athletics Track, Malta", "Good"),
        ("2025-05-10", 46.95, "Mary Peters Track, Belfast", "Good"),  # Season best!
        ("2025-05-24", 47.68, "Boudewijnstadion, Brussels", "Good"),
        ("2025-06-07", 47.37, "Putbosstadion, Oordegem, Belgium", "Good"),
        ("2025-06-22", 47.46, "Stratford Community Track, London", "Good"),
        ("2025-07-11", 48.02, "Morton Stadium, Santry, Dublin", "Good"),
        ("2025-08-02", 47.44, "Morton Stadium, Santry, Dublin", "Good"),
        ("2025-08-03", 47.85, "Morton Stadium, Santry, Dublin", "Good"),
    ]
    
    try:
        for date_str, time_sec, location, conditions in races:
            race = RaceResult(
                athlete_id=1, 
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                event="400m",
                time_seconds=time_sec,
                position=None,  # Position data not available
                location=location,
                conditions=conditions
            )
            db.add(race)
        
        db.commit()
        print(f"✅ Added {len(races)} real race results")
        print(f"   Season Best: 46.95s (May 10, Belfast)")
        print(f"   Latest Race: 47.85s (Aug 3)")
    
    finally:
        db.close()


def generate_training_data():
    """Generate realistic training data between races."""
    db = get_db()
    
    try:
        # Training period: February to August 2025
        start_date = datetime(2025, 2, 1)
        end_date = datetime(2025, 8, 3)
        
        current_date = start_date
        session_count = 0
        metric_count = 0
        
        # Training types and typical patterns
        training_types = ["Speed Endurance", "Tempo", "Speed", "Strength", "Recovery"]
        
        # Baseline metrics (based on elite 400m runner profile)
        base_hrv = 65
        base_rhr = 48
        base_sleep = 7.5
        
        while current_date <= end_date:
            # Train 5-6 days per week
            if random.random() < 0.83:  # ~5-6 days/week
                
                # Vary training type
                training_type = random.choice(training_types)
                
                # Session intensity by type
                intensity_map = {
                    "Speed": random.uniform(8.5, 10.0),
                    "Speed Endurance": random.uniform(8.0, 9.5),
                    "Tempo": random.uniform(6.5, 8.0),
                    "Strength": random.uniform(7.0, 8.5),
                    "Recovery": random.uniform(3.0, 5.5)
                }
                
                session = TrainingSession(
                    athlete_id=1,
                    date=current_date.date(),
                    session_type=training_type,
                    duration_minutes=random.randint(70, 120),
                    intensity_rpe=intensity_map[training_type],
                    notes=f"{training_type} session"
                )
                db.add(session)
                db.flush()
                
                session_count += 1
                
                # Add performance metrics (90% of sessions)
                if random.random() < 0.9:
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
            
            current_date += timedelta(days=1)
        
        db.commit()
        print(f"✅ Generated {session_count} training sessions")
        print(f"✅ Generated {metric_count} performance metrics")
    
    finally:
        db.close()


def update_athlete_info():
    """Update athlete profile with real information."""
    db = get_db()  # ADD THIS LINE
    
    try:
        athlete = db.query(Athlete).filter(Athlete.id == 1).first()
        if athlete:
            athlete.first_name = "Athlete"
            athlete.last_name = "one"
            athlete.personal_best_400m = 46.95  # All-time PB
            athlete.date_of_birth = datetime(2000, 1, 1)  # Anonymized
            athlete.gender = "Male"
            athlete.weight_kg = 86  
            athlete.height_cm = 190  # 
            
            db.commit()
            print("✅ Updated Athlete profile")
            print(f"   Personal Best: 46.95s")
    
    finally:
        db.close()


if __name__ == "__main__":
    print("🏃 Adding athlete real 2025 race data...\n")
    
    # Step 1: Clear old synthetic data
    clear_existing_data()
    
    # Step 2: Update athlete info
    update_athlete_info()
    
    # Step 3: Add real race results
    add_real_races()
    
    # Step 4: Generate realistic training data
    print("\n🏋️ Generating training data (Feb-Aug 2025)...")
    generate_training_data()
    
    print("\n" + "="*60)
    print("🎉 REAL DATA ADDED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Retrain models: python src/models.py")
    print("2. View dashboard: streamlit run src/dashboard.py")
    print("\nYour dashboard will now show:")
    print("- 8 real 400m races from 2025")
    print("- Season progression (46.95s PB in May)")
    print("- Realistic training patterns")


