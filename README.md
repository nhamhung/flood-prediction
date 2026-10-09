# Flood Prediction

A worked, end-to-end data science project built around Kaggle's
[Playground Series S4E5 — "Regression with a Flood Prediction Dataset"](https://www.kaggle.com/competitions/playground-series-s4e5):
predicting an area's **flood probability** from its scores on 20 flood-risk
factors (monsoon intensity, drainage, deforestation, urbanization, …).

**Best late submission: private R² 0.86893 — would have ranked #207 of 2,794
(top 7.4%)**, up from #535 with this project's first submission. See
[The leaderboard climb](#the-leaderboard-climb) for every step, including what
didn't help.

## The one finding that shapes everything

The competition data was synthetically generated from a 50,000-row original
dataset. In that original, **`FloodProbability = 0.005 × (sum of the 20
factors)` — exactly, for every row.** The synthetic generator added noise to
the factor values, so the real task is *recovering the original row sum from
noisy features*. That's why this project's features are row-level statistics
(sum, spread, skew, sorted values) rather than per-factor transformations,
and why SHAP attributes 93% of the model's decisions to `row_sum` alone.

## Two models, two jobs

| | App model | Leaderboard model |
|---|---|---|
| What | One LightGBM pipeline | LightGBM + XGBoost + CatBoost, 7-fold × 2 seeds, non-negative blend |
| Built by | `scripts/train.py` (~1 min) | `scripts/make_ensemble_submission.py --folds 7 --seeds 2` (~22 min) |
| Used by | Notebook, Streamlit app, SHAP, `scripts/make_submission.py` | Kaggle submissions |
| Why keep it | Loads instantly; SHAP explains every prediction | Best score (OOF R² 0.86943) |

## Prerequisites

Install once, before Setup below:

| Dependency | Why | Install |
|---|---|---|
| **Python 3.12** | This project's `.venv` is built against 3.12 — a different version may resolve incompatible package versions from `requirements.txt`. | [python.org/downloads](https://www.python.org/downloads/) or a version manager (e.g. `pyenv install 3.12`) |
| **Quarto** | Renders `report/report.qmd` — a standalone binary, not a Python package, so `pip install` never gets it. | [quarto.org/docs/get-started](https://quarto.org/docs/get-started/) |
| **Kaggle API token** | Needed to download the real dataset and leaderboards (not needed to run `pytest`, which uses synthetic data). | Kaggle account → **Account → Create New API Token** → save the downloaded file as `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows). See the [Kaggle API docs](https://www.kaggle.com/docs/api). |
| **Docker** (optional) | Only if you want to run the app in its pre-baked container instead of `streamlit run`. | [docker.com/get-started](https://www.docker.com/get-started/) |

## What's here

| Deliverable | Where |
|---|---|
| A well-documented notebook: EDA, the sum discovery, feature engineering, model comparison, SHAP, the leaderboard | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app: a flood-risk calculator with per-prediction SHAP, plus data/model/leaderboard pages | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup | `report/report.qmd` |
| Scripts: train the app model, build the leaderboard blend, make submissions, place a score on the final leaderboard | `scripts/` |

All of these share one source of truth in `src/flood_prediction/` —
`config.py` (paths, schema), `data.py` (loading), `features.py` (row
statistics), `model.py` (pipelines, CV, blending), `interpretability.py`
(SHAP), `leaderboard.py` ("would have ranked" placement + the submission
history) — so the notebook, app, scripts, and report can never quietly
drift apart.

## Project layout

```
data/raw/                    # Kaggle CSVs (gitignored) + data/raw/original/flood.csv
data/raw/leaderboard/        # frozen final public/private leaderboards (committed)
data/processed/              # ensemble_predictions.npz from the leaderboard run (gitignored)
notebooks/                   # the main EDA + modeling notebook
src/flood_prediction/        # shared config, data, features, models, SHAP, leaderboard placement
models/                      # app model artifact (model.joblib, gitignored)
app/                         # Streamlit app (multi-page, app/pages_src/) + Dockerfile
scripts/                     # train, make_submission, make_ensemble_submission, fetch_leaderboard, would_rank
report/                      # Quarto research writeup
tests/                       # pytest tests (synthetic data)
```

## Setup

```bash
python3.12 -m venv .venv          # use the 3.12 interpreter specifically
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

This project doesn't commit Kaggle's competition data (redistribution isn't
permitted). Accept the competition's rules on kaggle.com once, then:

```bash
# Competition data
kaggle competitions download -c playground-series-s4e5 -p data/raw
unzip -o data/raw/playground-series-s4e5.zip -d data/raw

# The 50k-row original dataset (for the "it's a sum" EDA)
kaggle datasets download naiyakhalid/flood-prediction-dataset -f flood.csv -p data/raw/original --unzip
```

The final leaderboards are already committed in `data/raw/leaderboard/`; to
refresh them, run `python scripts/fetch_leaderboard.py`.

## Run the notebook

```bash
python -m ipykernel install --user --name flood-prediction --display-name "Flood Prediction"   # once
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

It loads `models/model.joblib` (for SHAP) and
`data/processed/ensemble_predictions.npz` (for the blend section), so run
`scripts/train.py` and `scripts/make_ensemble_submission.py` first. Runs top
to bottom in about 3 minutes.

## Train and submit

```bash
python scripts/train.py                        # app model (LightGBM) -> models/model.joblib
python scripts/train.py --model xgboost        # any model in MODEL_FACTORIES (see --help)

python scripts/make_ensemble_submission.py --folds 7 --seeds 2   # leaderboard blend
kaggle competitions submit -c playground-series-s4e5 -f submission_ensemble.csv -m "message"

python scripts/would_rank.py                   # place your submissions on the final leaderboard
python scripts/would_rank.py --private 0.86899 # or any score
```

## The leaderboard climb

The competition closed on 31 May 2024, so these are late submissions —
scored by Kaggle, placed by `scripts/would_rank.py` against the frozen
final leaderboard.

| Step | Change | OOF R² | Public | Private | Would rank (private) |
|---|---|---|---|---|---|
| v1 | LightGBM + XGBoost + CatBoost averaged; row stats + sorted values; first-guess hyperparameters | 0.86916 | 0.86909 | 0.86870 | #535 (top 19%) |
| v2 | `colsample_bytree` 0.5 → 1.0, deeper XGBoost, NNLS blend weights | 0.86940 | 0.86931 | 0.86893 | **#207 (top 7.4%)** |
| v3 | 7 folds × 2 seeds | 0.86943 | 0.86933 | 0.86893 | #207 (top 7.4%) |

What mattered, in order:

1. **Row statistics** (+0.026 R² on a holdout) — the direct consequence of
   the target being a sum.
2. **Letting every tree see every column** (+0.0005) — with the signal
   concentrated in `row_sum`, sampling half the columns per tree means half
   the trees can't see it. This was the main change from v1 to v2, which
   moved the private rank from #535 to #207.

What didn't help (details in the report): adding the 50k original rows to
training (−0.0005 — they're noise-free, a different distribution), a
neural network (plateaued at 0.864, zero blend weight), value-count and
moment features, deeper CatBoost, slower learning rates.

**Why not top 50?** The top 500 private scores fit in a 0.0004 band. Every
team that scored ≥ 0.86899 private (top ~56) had ≥ 0.86939 public, and most
of them submitted the same public notebook that averaged *other people's
submission files* (including AutoGluon runs). Everything here is trained
from scratch instead, so the score reflects what this repo's method does.

## Run the app

A 4-page app: **Flood Risk Calculator** (20 sliders grouped by theme, or
load a random real area; shows the model's prediction next to the original
formula, plus a SHAP breakdown of that exact prediction), **Dataset
Overview** (the "it's a sum" discovery, interactively), **Model Insights**
(live SHAP + the blend's members and weights), and **Leaderboard Climb**.

```bash
streamlit run app/streamlit_app.py
```

Or in Docker (from the project root, after getting the data and training the app model):

```bash
docker build -t flood-prediction-app -f app/Dockerfile .
docker run -p 8501:8501 flood-prediction-app
```

Then open http://localhost:8501.

## Render the research writeup

`report/report.qmd` expects the `flood-prediction` Jupyter kernel (see
"Run the notebook"). Render with the project's venv active — or point
Quarto at it explicitly, since Quarto otherwise uses whichever `python3`
is first on your PATH:

```bash
source .venv/bin/activate
quarto render report/report.qmd                                  # HTML + PDF
QUARTO_PYTHON=.venv/bin/python quarto render report/report.qmd   # without activating
```

PDF needs a LaTeX distribution (`quarto install tinytex` once). The PDF
hides code cells; the HTML keeps them, folded.

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: No module named 'nbclient'` | Quarto is using a Python outside the venv — activate it or set `QUARTO_PYTHON` as above. |
| `Jupyter engine failed ... kernel not found` | The `ipykernel install --name flood-prediction` step hasn't been run yet. |
| `FileNotFoundError: ... model.joblib` / `ensemble_predictions.npz` | Run `scripts/train.py` / `scripts/make_ensemble_submission.py` first. |

## Run the tests

```bash
pytest tests/
```

33 tests covering the feature engineering, models, blending, seed bagging,
SHAP helpers, and leaderboard placement — all on synthetic data, no Kaggle
download needed.

## Deploy

The Streamlit app starts immediately from a bundled 5,000-row random sample sourced from Kaggle (plus the CC0 original dataset). Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete dataset through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/flood-prediction>
- Report: <https://nhamhung.github.io/flood-prediction/>
- Streamlit: <https://flood-prediction.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
