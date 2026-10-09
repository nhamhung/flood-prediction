"""Multi-page Streamlit app entry point.

Run locally:
    streamlit run app/streamlit_app.py

Or via Docker (from the project root):
    docker build -t flood-prediction-app -f app/Dockerfile .
    docker run -p 8501:8501 flood-prediction-app
"""

import streamlit as st

from pages_src import leaderboard_climb, model_insights, overview, predict

st.set_page_config(page_title="Flood Risk Predictor", page_icon="🌊", layout="wide")

pages = [
    # Every page module exports a same-named `render` function, so Streamlit's
    # default URL-pathname inference (from the callable's __name__) would
    # collide across all pages — explicit url_path avoids that.
    st.Page(predict.render, title="Flood Risk Calculator", icon="🌊", url_path="predict", default=True),
    st.Page(overview.render, title="Dataset Overview", icon="📊", url_path="overview"),
    st.Page(model_insights.render, title="Model Insights", icon="🧠", url_path="model-insights"),
    st.Page(leaderboard_climb.render, title="Leaderboard Climb", icon="🏆", url_path="leaderboard"),
]

navigation = st.navigation(pages)
navigation.run()
