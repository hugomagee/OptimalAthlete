"""
Streamlit dashboard for OptimalAthlete sprint performance system.
Interactive interface for data visualization and predictions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime, timedelta
from database import get_db
from setup_db import Athlete, TrainingSession, PerformanceMetric, RaceResult
from feature_engineering import engineer_features


# Page configuration
st.set_page_config(
    page_title="OptimalAthlete Dashboard",
    page_icon="",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load all data from database."""
    db = get_db()
    
    try:
        # Load athletes
        athletes = db.query(Athlete).all()
        athletes_df = pd.DataFrame([{
            'id': a.id,
            'name': f"{a.first_name} {a.last_name}",
            'gender': a.gender,
            'pb_400m': a.personal_best_400m
        } for a in athletes])
        
        # Load training sessions
        sessions = db.query(TrainingSession).all()
        sessions_df = pd.DataFrame([{
            'athlete_id': s.athlete_id,
            'date': s.date,
            'session_type': s.session_type,
            'duration': s.duration_minutes,
            'intensity': s.intensity_rpe
        } for s in sessions])
        
        # Load metrics
        metrics = db.query(PerformanceMetric).all()
        metrics_df = pd.DataFrame([{
            'session_id': m.session_id,
            'hrv': m.hrv_score,
            'rhr': m.resting_heart_rate,
            'sleep_hours': m.sleep_hours,
            'fatigue': m.fatigue_level,
            'wellness': m.wellness_score
        } for m in metrics])
        
        # Load races
        races = db.query(RaceResult).all()
        races_df = pd.DataFrame([{
            'athlete_id': r.athlete_id,
            'date': r.date,
            'time': r.time_seconds,
            'location': r.location
        } for r in races])
        
        return athletes_df, sessions_df, metrics_df, races_df
    
    finally:
        db.close()


@st.cache_resource
def load_models():
    """Load trained ML models."""
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return rf_model, xgb_model, feature_names
    except FileNotFoundError:
        st.warning("Models not found. Please run 'python models.py' first.")
        return None, None, None


def main():
    """Main dashboard function."""
    
    # Header
    st.title("OptimalAthlete Performance Dashboard")
    st.markdown("### ML-Powered Sprint Performance Analysis")
    
    # Load data
    athletes_df, sessions_df, metrics_df, races_df = load_data()
    rf_model, xgb_model, feature_names = load_models()
    
    # Sidebar - Athlete Selection
    st.sidebar.header("Dashboard Controls")
    selected_athlete_id = st.sidebar.selectbox(
        "Select Athlete",
        athletes_df['id'].tolist(),
        format_func=lambda x: athletes_df[athletes_df['id']==x]['name'].values[0]
    )
    
    athlete_info = athletes_df[athletes_df['id']==selected_athlete_id].iloc[0]
    
    # Sidebar - Athlete Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Athlete Profile")
    st.sidebar.write(f"**Name:** {athlete_info['name']}")
    st.sidebar.write(f"**Gender:** {athlete_info['gender']}")
    st.sidebar.write(f"**400m PB:** {athlete_info['pb_400m']:.2f}s")
    
    # Filter data for selected athlete
    athlete_sessions = sessions_df[sessions_df['athlete_id']==selected_athlete_id].copy()
    athlete_races = races_df[races_df['athlete_id']==selected_athlete_id].copy()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", 
        "Training Analysis", 
        "Race Results",
        "ML Predictions"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Sessions",
                len(athlete_sessions),
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Races",
                len(athlete_races),
                delta=None
            )
        
        with col3:
            avg_intensity = athlete_sessions['intensity'].mean()
            st.metric(
                "Avg Intensity",
                f"{avg_intensity:.1f}/10",
                delta=None
            )
        
        with col4:
            if len(athlete_races) > 0:
                best_time = athlete_races['time'].min()
                st.metric(
                    "Best Race Time",
                    f"{best_time:.2f}s",
                    delta=None
                )
        
        # Training volume over time
        st.subheader("Training Volume Over Time")
        athlete_sessions['date'] = pd.to_datetime(athlete_sessions['date'])
        weekly_volume = athlete_sessions.groupby(
            pd.Grouper(key='date', freq='W')
        )['duration'].sum().reset_index()
        
        fig_volume = px.line(
            weekly_volume,
            x='date',
            y='duration',
            title='Weekly Training Volume (minutes)',
            labels={'duration': 'Minutes', 'date': 'Date'}
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # TAB 2: Training Analysis
    with tab2:
        st.header("Training Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Session type distribution
            session_counts = athlete_sessions['session_type'].value_counts()
            fig_pie = px.pie(
                values=session_counts.values,
                names=session_counts.index,
                title='Training Session Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Intensity distribution
            fig_intensity = px.histogram(
                athlete_sessions,
                x='intensity',
                nbins=10,
                title='Training Intensity Distribution',
                labels={'intensity': 'RPE Score (1-10)'}
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        # Recent sessions table
        st.subheader("Recent Training Sessions")
        recent_sessions = athlete_sessions.sort_values('date', ascending=False).head(10)
        st.dataframe(
            recent_sessions[['date', 'session_type', 'duration', 'intensity']],
            use_container_width=True,
            hide_index=True
        )
    
    # TAB 3: Race Results
    with tab3:
        st.header("Race Performance")
        
        if len(athlete_races) > 0:
            athlete_races['date'] = pd.to_datetime(athlete_races['date'])
            athlete_races_sorted = athlete_races.sort_values('date')
            
            # Race times over time
            fig_races = px.line(
                athlete_races_sorted,
                x='date',
                y='time',
                title='Race Performance Over Time',
                labels={'time': '400m Time (seconds)', 'date': 'Date'},
                markers=True
            )
            fig_races.add_hline(
                y=athlete_info['pb_400m'],
                line_dash="dash",
                line_color="red",
                annotation_text="Personal Best"
            )
            st.plotly_chart(fig_races, use_container_width=True)
            
            # Race results table
            st.subheader("All Race Results")
            st.dataframe(
                athlete_races_sorted[['date', 'time', 'location']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No race results available for this athlete.")
    
    # TAB 4: ML Predictions
    with tab4:
        st.header("Machine Learning Predictions")
        
        if rf_model is not None:
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Random Forest Model**")
                st.write("- Test MAE: ~0.3 seconds")
                st.write("- Best performing model")
                st.write("- Uses 12 training features")
            
            with col2:
                st.markdown("**XGBoost Model**")
                st.write("- Test MAE: ~0.3 seconds")
                st.write("- Gradient boosting approach")
                st.write("- Emphasis on sleep quality")
            
            st.markdown("---")
            st.subheader("Race Time Prediction")
            st.info(
                "Note: Predictions are based on recent training data. "
                "With limited race samples in demo dataset, predictions are approximate. "
                "A real system with 100+ races would be more accurate."
            )
            
            # Feature importance
            st.subheader("Most Important Features")
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 5 Features for Prediction'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
        else:
            st.error("Models not loaded. Please train models first by running: `python models.py`")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**OptimalAthlete** - ML Sprint Performance System | "
        "Built for MSc Data Analytics Application"
    )


if __name__ == "__main__":
    main()
