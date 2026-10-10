# Customer Churn Intelligence System

Capstone project for the Bluemind Solutions AI and ML internship, built over Days 18 to 20 by Abhishek Rai A. The full day-by-day work from the internship is in a separate repository: https://github.com/Abhirai2006/bluemind-ml-internship-2026

## Problem

A telecom provider loses roughly a quarter of its customers, and by the time someone cancels it is too late to do anything about it. This project builds a model that flags customers who are likely to leave, explains why each one is at risk, and estimates what a targeted retention effort would be worth.

## Data

IBM's Telco Customer Churn dataset: 7,043 customers and 21 columns covering account details, subscribed services, billing and whether the customer churned. The notebooks load it from a public GitHub mirror, so there is nothing to download by hand.

## Approach

1. Clean the data. The blank `TotalCharges` values belong to brand-new customers and are filled from monthly charges times tenure.
2. Run a short EDA on contract type, tenure and monthly charges, which explain most of the churn.
3. Build a leak-free preprocessing pipeline with `ColumnTransformer`, splitting before anything is fitted.
4. Train a logistic regression baseline, then tune Random Forest, XGBoost and LightGBM with `RandomizedSearchCV` and 5-fold stratified cross-validation, scored on ROC-AUC.
5. Choose the decision threshold from the precision-recall curve instead of using 0.5, since missing a real churner costs more than a wasted retention offer.
6. Explain the winning model with SHAP, globally and for single customers.
7. Review the ten worst predictions.
8. Estimate the retention opportunity in dollars.
9. Serve the model through a Streamlit dashboard that scores one customer or a whole CSV.

## Results

| Model | Cross-validated ROC-AUC |
|---|---|
| LightGBM (final model) | 0.849 |
| Random Forest | 0.848 |
| XGBoost | 0.848 |
| Logistic Regression (baseline) | 0.846 |

On the held-out test set the final model has a ROC-AUC of 0.847. At the chosen threshold it catches 82 percent of the customers who churn, at the cost of some false alarms, which suits a retention use case. The strongest churn drivers, in order, are month-to-month contracts, short tenure, no online security, high monthly charges, no tech support and paying by electronic check.

The test set has about $42,700 of monthly revenue at risk. With an assumed 35 percent retention success rate, that is close to $15,000 recoverable a month, or about $179,000 a year at this customer base size. The 35 percent is an assumption, not a measured figure.

## Repository layout

```
notebooks/
  01-data-cleaning-eda-baseline.ipynb           Day 18
  02-tuning-explainability-and-dashboard.ipynb  Day 19
app/
  app.py                  Streamlit dashboard
  churn_model.pkl         final LightGBM pipeline
  model_meta.pkl          model name and decision threshold
  requirements.txt        packages for deploying this folder
  .streamlit/config.toml  forces the light theme
sample_data/              two small CSVs for trying the batch upload
sample_chart_outputs/     charts produced by the notebooks, plus dashboard screenshots
presentation/             the 20-slide final presentation
requirements.txt          packages for the notebooks
```

## How to run

Open the two notebooks in Google Colab, in order. They run top to bottom. The last cells of the second notebook start the dashboard and print a public link through a `cloudflared` tunnel.

To run the dashboard on your own machine with Python 3.10 or newer:

```
pip install -r requirements.txt
streamlit run app/app.py
```

The batch tab takes a CSV with the 19 customer columns. The files in `sample_data/` show the format. `sample_batch_with_noise.csv` contains impossible values, such as a negative tenure, on purpose. The dashboard rejects those rows and lists the reason for each one instead of scoring them.

## Notes on the dashboard

- `app/.streamlit/config.toml` forces the light theme. Without it, Streamlit follows the viewer's system setting and the app can look different from the slides.
- The batch tab fixes `TotalCharges` the same way the notebooks do, because exported data sometimes has a blank there for customers who have not been billed yet. Extra spaces and different capitalisation in category columns are also corrected before scoring.
- The model files are pickles made with a specific scikit-learn and LightGBM version. If loading fails on another machine, install the versions in `requirements.txt` or rerun the second notebook to rebuild them.
- The dashboard screenshots in `sample_chart_outputs/` (files 12 to 14) show an earlier layout of the app.
- The notebook tunnels with `cloudflared` instead of `localtunnel`, since localtunnel's click-through page can block Streamlit's background requests and show a "Failed to fetch dynamically imported module" error.

## Hosting it permanently

Colab links stop working when the runtime disconnects. For a permanent link, push the contents of `app/` to a GitHub repository and deploy it on Streamlit Community Cloud with `app.py` as the main file. The two `.pkl` files must be committed too, because the app loads them at startup. Vercel is not a good fit for this app, since Streamlit needs a long-running process with a WebSocket connection and Vercel runs short serverless functions.

## What I would do with more time

Use a time-based validation split instead of a random one, since churn depends on time and a random split can leak information from the future. Try a survival-style model that predicts when a customer will leave and not only whether. Score the whole customer base on a weekly schedule and send the high-risk list to the retention team's CRM instead of using a manual CSV upload.
