# OptimalAthlete - ML Sprint Performance System

An intelligent sprint performance analysis system using machine learning to predict 400m race times and optimize training recommendations for elite athletes.

## Project Overview

OptimalAthlete analyzes training data, recovery metrics, and race performance to provide:
- **Performance Prediction**: ML models predict race times based on training history
- **Training Optimization**: Personalized recommendations to improve performance
- **Recovery Monitoring**: Track wellness metrics and prevent overtraining
- **Interactive Dashboard**: Real-time visualization of athlete data

## Target Users

Elite 400m sprinters, coaches, and sports scientists seeking data-driven performance insights.

##  Key Features

### 1. Data Management
- SQLite database storing athlete profiles, training sessions, performance metrics, and race results
- Synthetic data generation for model training and testing
- Scalable schema supporting multiple athletes

### 2. Machine Learning Models
- **Random Forest Regressor**: Predicts 400m race times
- **XGBoost Model**: Advanced gradient boosting predictions
- **Feature Engineering**: Training load, recovery scores, performance trends

### 3. Interactive Dashboard
- Built with Streamlit for real-time data visualization
- Athlete performance tracking
- Training recommendations
- Model performance metrics

## Technology Stack

- **Python 3.13**: Core programming language
- **SQLAlchemy**: Database ORM
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **Streamlit**: Dashboard interface
- **Plotly**: Interactive visualizations
- **SQLite**: Lightweight database

##  Project Structure
```
OptimalAthlete/
│
├── data/                          # Database and datasets
│   └── optimalathlete.db         # SQLite database
│
├── src/                           # Source code
│   ├── database.py               # Database connection management
│   ├── data_loader.py            # Synthetic data generation
│   ├── feature_engineering.py    # Feature creation for ML
│   ├── models.py                 # ML model training
│   └── dashboard.py              # Streamlit dashboard
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── docs/                          # Documentation
│   ├── figures/                  # Charts and visualizations
│   └── project_report.md         # Detailed project report
│
├── setup_db.py                    # Database schema definition
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start Guide

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
# Create database and tables
python setup_db.py
```

### 3. Generate Synthetic Data
```bash
# Populate database with training data
python src/data_loader.py
```

### 4. Train ML Models
```bash
# Train Random Forest and XGBoost models
python src/models.py
```

### 5. Launch Dashboard
```bash
# Start Streamlit dashboard
streamlit run src/dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Model Performance

- **Random Forest**: MAE ~0.3 seconds on 400m predictions
- **XGBoost**: MAE ~0.25 seconds with optimized hyperparameters
- **Feature Importance**: Training intensity, recovery metrics, recent performance trends

## Academic Context

This project was developed as part of my application to ISEG's MSc in Data Analytics for Business. It demonstrates:
- End-to-end ML pipeline development
- Database design and management
- Statistical modeling and feature engineering
- Data visualization and dashboard creation
- Real-world application of analytics to sports science

## Author

**Hugo Magee**
- Final-year Science student, University College Dublin
- Elite 400m sprinter competing for Ireland (PB: 46.9s)
- Target: MSc Data Analytics / Business analytics at ISEG (2025)

## Future Enhancements

- Integration with real training data via Garmin/Strava APIs
- Deep learning models (LSTM) for time-series predictions
- Mobile app for real-time data entry
- Multi-event support (100m, 200m, 800m)
- Coach collaboration features

## Contact

For questions or collaboration: [Hugo.magee1@ucdconnect.ie]

## License

This project is for academic and personal use.

---

**Built with by an athlete, for athletes**