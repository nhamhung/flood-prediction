# Build from the project root:
#   docker build -t flood-prediction-app -f app/Dockerfile .
# Run:
#   docker run -p 8501:8501 flood-prediction-app

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
# The app reads the training table directly (Dataset Overview's live EDA,
# the calculator's "random real area", Model Insights' live SHAP) plus the
# leaderboard snapshot and the original dataset — download them first, see
# the README's "Get the data" section. data/processed/ always exists (it holds a
# .gitkeep); ensemble_predictions.npz inside it is optional — Model Insights
# shows the blend breakdown only if scripts/make_ensemble_submission.py ran.
COPY data/raw/train.csv data/raw/train.csv
COPY data/raw/original/ data/raw/original/
COPY data/raw/leaderboard/ data/raw/leaderboard/
COPY data/processed/ data/processed/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
