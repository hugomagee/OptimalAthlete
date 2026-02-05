"""
Database connection and session management for OptimalAthlete.
Provides SQLAlchemy engine and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
import os

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'optimalathlete.db')
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# Create engine with appropriate settings for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False},  # Needed for SQLite
    poolclass=StaticPool,
    echo=False  # Set to True for SQL query logging during development
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create thread-safe session
Session = scoped_session(SessionLocal)


def get_db():
    """
    Get database session.
    
    Usage:
        db = get_db()
        try:
            # Do database operations
            result = db.query(Athlete).all()
        finally:
            db.close()
    
    Returns:
        SQLAlchemy session object
    """
    db = Session()
    try:
        return db
    except Exception as e:
        db.close()
        raise e


def init_database():
    """
    Initialize database by creating all tables.
    This imports Base from setup_db and creates tables.
    """
    from setup_db import Base
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized at: {DATABASE_PATH}")


def close_db():
    """
    Close all database sessions.
    Call this when shutting down the application.
    """
    Session.remove()
    print("✅ Database sessions closed")


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    init_database()
    
    # Try to get a session
    db = get_db()
    print("✅ Database connection successful!")
    db.close()