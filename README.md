# Online Shoppers Purchase Prediction

Data Mining (AIE323) final project, Alamein International University.
Project 6: E-Commerce Purchase Prediction and Behavior Mining.

We use the UCI Online Shoppers Purchasing Intention dataset to predict if a browsing session ends with a purchase, and K-Means to group sessions by browsing behavior. The trained models are used in a Streamlit app where you can score one session, upload a CSV of sessions, and look at the segments and patterns we found.

Live app: <add link after deploying>

## Team

| # | Name | Student ID |
|---|------|------------|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |

## Dataset

Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository (id 468):
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

12,330 sessions, 17 input columns and the target `Revenue` (15.5% of sessions are purchases). The CSV is included in `data/`.

## Folders

```
app.py                  Streamlit app
features.py             feature engineering, used by the notebook and the app
notebooks/              the full analysis notebook (EDA, preprocessing, models, tuning, K-Means)
models/                 saved pipelines, model_info.json, test set predictions
data/                   dataset and a small sample file for the app
figures/                all figures from the notebook
report/                 project report (docx and pdf)
presentation/           slides
requirements.txt        packages needed to run the app
requirements-dev.txt    extra packages to re-run the notebook
```

## Run the app locally

Python 3.12 was used.

```
python -m venv .venv
.venv\Scripts\activate          (on Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Re-run the notebook

```
pip install -r requirements-dev.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/online_shoppers_project.ipynb
```

Running the notebook trains everything again and overwrites the files in `models/` and `figures/`. It takes around 10 minutes, most of it is the hyperparameter search.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "Create app", choose the repository, branch `main` and main file `app.py`.
4. Under advanced settings pick Python 3.12, then deploy.
5. Copy the app link into the "Live app" line at the top of this file and into the report.

## Results

Final model: random forest with balanced class weights, tuned with RandomizedSearchCV, decision threshold 0.632 (chosen from out-of-fold predictions on the training set).

| Metric (test set) | Value |
|---|---|
| ROC-AUC | 0.933 |
| PR-AUC | 0.742 |
| Precision | 0.668 |
| Recall | 0.717 |
| F1 | 0.692 |
| False alarm rate | 0.066 |

The baseline (always predicting the buyer rate) has a PR-AUC of 0.156. Details for all models are in the notebook and in the "Model performance" tab of the app.

K-Means found 5 session segments, from one-page bouncers (0.5% buy) to sessions that reach pages with a page value (66% buy).

The predictions are estimates based on past sessions from one shop. They are not certain and should not be the only reason for a decision.
