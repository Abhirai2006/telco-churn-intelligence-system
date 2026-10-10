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

