"""
Streamlit dashboard for OptimalAthlete.

Presents the walk-forward evaluation as the headline result and shows the
naive pooled split beside it, struck through, as the leakage cautionary tale.
Visual styling lives in theme.py.
"""

import json
import os
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

import theme
from database import get_db
from setup_db import Athlete, PerformanceMetric, RaceResult, TrainingSession

st.set_page_config(
    page_title="OptimalAthlete",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(theme.CSS, unsafe_allow_html=True)


@st.cache_resource
def ensure_data_and_models():
    """Bootstrap the database and models on first run (e.g. Streamlit Cloud).

    Creates the SQLite schema, generates synthetic demo data if the database
    is empty, and trains models if no saved models exist yet.
    """
    from database import init_database
    init_database()

    db = get_db()
    try:
        if db.query(Athlete).count() == 0:
            from data_loader import generate_synthetic_data
            generate_synthetic_data(db, num_athletes=5, days_of_data=180)
    finally:
        db.close()

    if not os.path.exists('models/random_forest_model.pkl'):
        from models import train_models
        train_models()


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


@st.cache_data
def load_model_metrics():
    """Load evaluation metrics saved at training time, if available."""
    try:
        with open('models/model_metrics.json') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def render_protocol_comparison(metrics: dict) -> None:
    """Walk-forward as the headline; the pooled split struck through beside it."""
    walk = (metrics or {}).get("walk_forward") or {}
    naive = (metrics or {}).get("naive_pooled_split") or {}

    if not walk:
        st.info(
            "No walk-forward metrics saved yet. Run `python models.py` to generate them."
        )
        return

    st.markdown("### Reported protocol — walk-forward validation")
    st.markdown(
        '<div class="note">Races are sorted by date and each one is predicted by a '
        'model fitted <strong>only on races that happened before it</strong>. This is '
        'the only protocol that answers the question the system poses: given training '
        'data up to today, what will the next race time be?</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for key in ("random_forest", "xgboost", "baseline_recent_average"):
        m = walk.get(key)
        if not m:
            continue
        rows.append({
            "Model": m["model_name"],
            "MAE (s)": round(m["mae"], 3),
            "RMSE (s)": round(m["rmse"], 3),
            "R²": round(m["r2"], 3) if m.get("r2") is not None else None,
            "Races predicted": m["n_predictions"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    baseline = walk.get("baseline_recent_average")
    models = {k: v for k, v in walk.items() if k != "baseline_recent_average"}
    if baseline and models:
        best_key = min(models, key=lambda k: models[k]["mae"])
        best = models[best_key]
        if best["mae"] < baseline["mae"]:
            verdict = (
                f"<strong>The best model beats the baseline.</strong> "
                f"{best['model_name']} reaches MAE {best['mae']:.3f}s against the "
                f"baseline's {baseline['mae']:.3f}s."
            )
        else:
            verdict = (
                f"<strong>No model beats the baseline.</strong> The best model "
                f"({best['model_name']}, MAE {best['mae']:.3f}s) is worse than simply "
                f"predicting each athlete's recent average ({baseline['mae']:.3f}s). "
                f"On this data the models have not demonstrated predictive value — "
                f"which is the correct answer, because the bundled demo data contains "
                f"no relationship between training features and race time by construction."
            )
        st.markdown(f'<div class="note">{verdict}</div>', unsafe_allow_html=True)

    st.markdown("### Cautionary comparison — naive pooled split")
    st.markdown(
        '<div class="note">These are the numbers this project used to report, and they '
        'are <strong>not a result</strong>. A random train/test split over pooled '
        'athletes leaks in two directions: future races inform predictions of past '
        'ones, and the same athlete appears on both sides of the split — and most of '
        'the variance in pooled race times is <em>between</em> athletes rather than '
        'within them. The R² below is what that leakage buys you on data with no real '
        'signal in it.</div>',
        unsafe_allow_html=True,
    )

    for key in ("random_forest", "xgboost"):
        m = naive.get(key)
        if not m:
            continue
        st.markdown(
            f'<div class="row"><span class="row-key">{m["model_name"]} '
            f'— pooled-split test R²</span>'
            f'<span class="row-val retracted">{m["test_r2"]:.3f}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="note">An earlier version of this project claimed R² = 0.84 on '
        'personal training data. The pooled split reproduces a figure of that '
        'magnitude on synthetic data built to contain no signal whatsoever — which is '
        'the clearest possible demonstration that the original number measured the '
        'protocol, not the athlete.</div>',
        unsafe_allow_html=True,
    )


# ── sections ──────────────────────────────────────────────────────────────

def render_overview(athlete_sessions: pd.DataFrame, athlete_races: pd.DataFrame) -> None:
    st.markdown("## Key figures")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", f"{len(athlete_sessions):,}")
    col2.metric("Races", f"{len(athlete_races):,}")
    col3.metric("Avg intensity", f"{athlete_sessions['intensity'].mean():.1f}")
    if len(athlete_races):
        col4.metric("Best 400m", f"{athlete_races['time'].min():.2f}s")

    st.markdown("## Training volume")
    st.markdown(
        '<p class="lede">Total session minutes per week</p>', unsafe_allow_html=True
    )
    sessions = athlete_sessions.copy()
    sessions['date'] = pd.to_datetime(sessions['date'])
    weekly = (
        sessions.groupby(pd.Grouper(key='date', freq='W'))['duration']
        .sum().reset_index()
    )
    fig = px.line(weekly, x='date', y='duration')
    fig.update_traces(
        line=dict(color=theme.SERIES[0], width=2),
        hovertemplate='%{x|%d %b %Y}<br>%{y:,.0f} min<extra></extra>',
    )
    theme.style_figure(fig, y_title='minutes per week', height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_training(athlete_sessions: pd.DataFrame) -> None:
    st.markdown("## Session mix")
    st.markdown(
        '<p class="lede">Count of sessions by type</p>', unsafe_allow_html=True
    )
    counts = athlete_sessions['session_type'].value_counts().sort_values()
    fig = px.bar(x=counts.values, y=counts.index, orientation='h')
    fig.update_traces(
        marker_color=theme.SERIES[0],
        hovertemplate='%{y}<br>%{x} sessions<extra></extra>',
    )
    theme.style_figure(fig, x_title='sessions', height=280)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Session intensity")
    st.markdown(
        '<p class="lede">Distribution of rate-of-perceived-exertion scores</p>',
        unsafe_allow_html=True,
    )
    fig = px.histogram(athlete_sessions, x='intensity', nbins=24)
    fig.update_traces(
        marker_color=theme.SERIES[0],
        hovertemplate='RPE %{x}<br>%{y} sessions<extra></extra>',
    )
    theme.style_figure(fig, x_title='RPE (1–10)', y_title='sessions', height=280)
    fig.update_layout(bargap=0.08)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Recent sessions")
    recent = athlete_sessions.sort_values('date', ascending=False).head(12)
    table = pd.DataFrame({
        'Date': pd.to_datetime(recent['date']).dt.strftime('%Y-%m-%d'),
        'Type': recent['session_type'],
        'Minutes': recent['duration'],
        'RPE': recent['intensity'].round(1),
    })
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_races(athlete_races: pd.DataFrame, athlete_info) -> None:
    if len(athlete_races) == 0:
        st.info("No race results available for this athlete.")
        return

    st.markdown("## Race times")
    st.markdown(
        '<p class="lede">400m times over the recorded period. '
        'The axis is inverted so that faster is higher.</p>',
        unsafe_allow_html=True,
    )
    races = athlete_races.copy()
    races['date'] = pd.to_datetime(races['date'])
    ordered = races.sort_values('date')

    fig = px.line(ordered, x='date', y='time', markers=True)
    fig.update_traces(
        line=dict(color=theme.SERIES[0], width=2),
        marker=dict(size=7, line=dict(width=2, color=theme.SURFACE)),
        hovertemplate='%{x|%d %b %Y}<br>%{y:.2f}s<extra></extra>',
    )
    fig.add_hline(
        y=athlete_info['pb_400m'],
        line_dash="dot", line_color=theme.MUTED, line_width=1,
        annotation_text="personal best", annotation_position="top left",
        annotation_font=dict(size=11, color=theme.MUTED),
    )
    theme.style_figure(fig, y_title='seconds', reverse_y=True, height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## All races")
    table = pd.DataFrame({
        'Date': ordered['date'].dt.strftime('%Y-%m-%d'),
        'Time (s)': ordered['time'].round(2),
        'Location': ordered['location'],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_models(rf_model, feature_names) -> None:
    st.markdown("## How these models are evaluated")
    metrics = load_model_metrics()
    render_protocol_comparison(metrics)

    if rf_model is None or not feature_names:
        st.error("Models not loaded. Run `python models.py` first.")
        return

    st.markdown("## Feature importance")
    st.markdown(
        '<p class="lede">Random Forest impurity importance. Importance describes what '
        'the model leaned on, not what causes race times — on data with no true signal '
        'it reflects noise.</p>',
        unsafe_allow_html=True,
    )
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_,
    }).sort_values('Importance').tail(8)

    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
    fig.update_traces(
        marker_color=theme.SERIES[0],
        hovertemplate='%{y}<br>%{x:.3f}<extra></extra>',
    )
    theme.style_figure(fig, x_title='importance', height=300)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    if metrics:
        st.markdown(
            f'<div class="note"><strong>Dataset.</strong> '
            f'{metrics.get("n_samples", "?")} races, '
            f'{metrics.get("n_features", len(feature_names))} engineered features. '
            f'Rolling features use genuine calendar windows (7 and 14 days), not row '
            f'counts.</div>',
            unsafe_allow_html=True,
        )


# Sections in tab order. The keys double as the values accepted by
# OA_ONLY_SECTION, which tools/capture_screenshots.py sets so each screenshot in
# docs/screenshots/ can be regenerated from the live app one section at a time.
SECTIONS = ("overview", "training", "races", "models")
SECTION_LABELS = {
    "overview": "Overview",
    "training": "Training analysis",
    "races": "Race results",
    "models": "Model evaluation",
}


def main():
    """Main dashboard function."""
    st.markdown("# OptimalAthlete")
    st.markdown(
        '<p class="lede">An n=1 training-data measurement methodology · '
        'CI-tested · ready for real wearable data</p>'
        '<span class="kicker">Synthetic demo data — no signal by construction</span>',
        unsafe_allow_html=True,
    )

    ensure_data_and_models()
    athletes_df, sessions_df, metrics_df, races_df = load_data()
    rf_model, xgb_model, feature_names = load_models()

    st.sidebar.markdown("## Controls")
    selected_athlete_id = st.sidebar.selectbox(
        "Athlete",
        athletes_df['id'].tolist(),
        format_func=lambda x: athletes_df[athletes_df['id'] == x]['name'].values[0],
    )
    athlete_info = athletes_df[athletes_df['id'] == selected_athlete_id].iloc[0]

    st.sidebar.markdown("## Profile")
    st.sidebar.markdown(
        f'<div class="row"><span class="row-key">Name</span>'
        f'<span class="row-val">{athlete_info["name"]}</span></div>'
        f'<div class="row"><span class="row-key">Gender</span>'
        f'<span class="row-val">{athlete_info["gender"]}</span></div>'
        f'<div class="row"><span class="row-key">400m PB</span>'
        f'<span class="row-val">{athlete_info["pb_400m"]:.2f}s</span></div>',
        unsafe_allow_html=True,
    )

    athlete_sessions = sessions_df[
        sessions_df['athlete_id'] == selected_athlete_id
    ].copy()
    athlete_races = races_df[races_df['athlete_id'] == selected_athlete_id].copy()

    # Docs mode: render one section without the tab chrome so each screenshot
    # can be captured at full height.
    only = os.environ.get("OA_ONLY_SECTION")
    if only in SECTIONS:
        {
            "overview": lambda: render_overview(athlete_sessions, athlete_races),
            "training": lambda: render_training(athlete_sessions),
            "races": lambda: render_races(athlete_races, athlete_info),
            "models": lambda: render_models(rf_model, feature_names),
        }[only]()
    else:
        tabs = st.tabs([SECTION_LABELS[s] for s in SECTIONS])
        with tabs[0]:
            render_overview(athlete_sessions, athlete_races)
        with tabs[1]:
            render_training(athlete_sessions)
        with tabs[2]:
            render_races(athlete_races, athlete_info)
        with tabs[3]:
            render_models(rf_model, feature_names)

    st.markdown(
        '<div class="footer"><strong>OptimalAthlete</strong> — an n=1 measurement '
        'methodology for training and race data. All figures on this page are computed '
        'from the bundled synthetic database, which is generated from a fixed seed and '
        'contains no relationship between training features and race outcomes. '
        'Nothing here is a performance claim.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
