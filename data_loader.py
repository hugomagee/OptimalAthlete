"""
Synthetic data generator for OptimalAthlete sprint performance system.

IMPORTANT — what this data does and does not contain.

Race times here are generated as ``personal_best + uniform(-0.3, 1.5)``: they
depend on **the athlete's identity and nothing else**. There is deliberately no
relationship between any training or recovery feature and the race outcome.

That makes this dataset a test of the *evaluation protocol* rather than of the
models. The correct result on data with no signal is that a model fails to beat
a "predict this athlete's recent average" baseline. Walk-forward validation
recovers exactly that. The naive pooled random split, by contrast, reports a
positive R² — which is pure leakage, since the only real structure in the data
is athlete identity and a pooled split lets the model see each athlete on both
sides. See ``models.py`` and the README's Model Performance section.

Generation is seeded and dates are anchored to a fixed reference date, so a
fresh clone reproduces the same database and therefore the same metrics.
"""

import random
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from setup_db import Athlete, PerformanceMetric, RaceResult, TrainingSession

# Fixed anchor so generated dates — and every downstream metric — are
# reproducible. Previously this used datetime.now(), which meant a fresh clone
# produced a different database and different numbers on every run.
REFERENCE_DATE = datetime(2025, 6, 30)
DEFAULT_SEED = 42


def generate_synthetic_data(
    db: Session,
    num_athletes: int = 5,
    days_of_data: int = 540,
    seed: int = DEFAULT_SEED,
    reference_date: datetime = REFERENCE_DATE,
) -> None:
    """
    Generate synthetic training data for sprint athletes.

    Args:
        db: SQLAlchemy database session
        num_athletes: Number of athletes to generate (default 5)
        days_of_data: Number of days of historical data (default 540)
        seed: RNG seed — fixed so the generated database is reproducible
        reference_date: end date of the generated window (fixed, not "today")
    """
    random.seed(seed)
    print(f"Generating synthetic data for {num_athletes} athletes over {days_of_data} days...")

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
            date_of_birth=datetime(
                2000 + random.randint(0, 4), random.randint(1, 12), random.randint(1, 28)
            ),
            gender="Male" if i % 2 == 0 else "Female",
            personal_best_400m=pb,
            weight_kg=65 + random.randint(-10, 10),
            height_cm=170 + random.randint(-10, 10)
        )
        db.add(athlete)
        athletes.append(athlete)

    db.commit()
    print(f"Created {num_athletes} athletes")

    # Generate training sessions for each athlete
    start_date = reference_date - timedelta(days=days_of_data)
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

        while current_date <= reference_date:
            # Training happens 5-6 days per week
            if random.random() < 0.8:
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
                db.flush()

                session_count += 1

                # Add performance metrics (collected most days)
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

                # Add race results (approximately every 2-3 weeks during season)
                if training_type == "Race Pace" and random.random() < 0.45:
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

    print(f"Created {session_count} training sessions")
    print(f"Created {metric_count} performance metrics")
    print(f"Created {race_count} race results")
    print("Synthetic data generation complete!")


if __name__ == "__main__":
    from database import get_db, init_database

    # Initialize database
    init_database()

    # Generate data
    db = get_db()
    try:
        generate_synthetic_data(db, num_athletes=5, days_of_data=540)
    finally:
        db.close()
