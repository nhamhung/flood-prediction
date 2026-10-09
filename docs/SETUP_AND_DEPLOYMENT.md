# Setup and Deployment

## Local setup

```bash
git clone https://github.com/nhamhhung/flood-prediction.git
cd flood-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r app/requirements.txt
pytest tests/
python scripts/smoke_app.py
streamlit run app/streamlit_app.py
```

## Deploy your fork

1. Fork or clone this repository and replace the values in `docs/deployment-config.yml`.
2. Push to your own public GitHub repository with `main` as the default branch.
3. In Settings → Pages, select **GitHub Actions** as the source.
4. In Streamlit Community Cloud, create a new app from your repository, branch `main`, file `app/streamlit_app.py`.
5. The bundled Kaggle-derived sample needs no secret. To use full data, set `USE_FULL_KAGGLE_DATA=true`, configure either `KAGGLE_API_TOKEN` or a `[kaggle]` section with username and key, and accept the playground-series-s4e5 competition rules first.
6. Wait for CI and Pages to pass, then record the URLs and revision in `docs/DEPLOYMENT_ACCEPTANCE.md`.

Do not copy another owner's tokens, application URL, Pages URL, or secret values. GitHub Pages hosts static documentation; Streamlit Community Cloud runs the Python app.

## Required checks

Protect `main` after the initial bootstrap. Require `ruff`, `pytest (3.11)`, `pytest (3.12)`, `smoke (3.11)`, `smoke (3.12)`, and `pages-build`. Disable force pushes and direct pushes; use up-to-date pull requests and squash merges.

## Rollback

Use the most recent SHA recorded as known-good in `DEPLOYMENT_ACCEPTANCE.md`. Revert later commits on a `rollback/<date>` branch, pass all required checks, and merge normally. Verify both Pages and Streamlit before closing the rollback.
