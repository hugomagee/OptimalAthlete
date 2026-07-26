# Deploying to Streamlit Community Cloud (free)

The app is self-bootstrapping: on first run it creates the SQLite database,
generates synthetic demo data, and trains the models automatically. No manual
setup steps are needed on the server.

## Steps

1. Push this repository to GitHub (public repos deploy free).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository**: `hugomagee/OptimalAthlete`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
5. Under **Advanced settings**, select **Python 3.12**.
6. Click **Deploy**.

The first load takes a couple of minutes while dependencies install and the
app generates data and trains models. Subsequent loads are fast.

## Configuration files used

- `requirements.txt` — pinned dependencies installed by Streamlit Cloud
- `.streamlit/config.toml` — dark theme and telemetry settings

## Notes

- Streamlit Community Cloud has an ephemeral filesystem: the SQLite database
  and trained models are regenerated whenever the app container restarts.
  That is fine for the synthetic demo data this repo ships with.
- To use real training data in a deployment, store it in a hosted database
  (e.g. Postgres on Supabase/Neon) and point `DATABASE_URL` in `database.py`
  at it via [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
