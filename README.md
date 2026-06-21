# GLWP-PDMD Model Training Code

This repository contains the XGBoost model training code used to reconstruct midday leaf water potential (Ψ_MD).

## Input data

The input file should be an Excel file containing the following columns:

- T
- NDVI
- VPD
- SM1
- ST1
- SM2
- ST2
- Sand
- Silt
- Clay
- md
- Station_ID
- date

`md` is the target variable, representing midday leaf water potential in MPa.

## Method

The script uses site-date grouped data splitting to avoid information leakage. Hyperparameters are optimized using Optuna with GroupKFold cross-validation. The final model is evaluated using a 20% hold-out test set and then retrained using the full dataset for global inversion.

## Output

The script saves two XGBoost models:

- `xgb_model_seed42_paper_unweighted.json`
- `xgb_model_FULL_DATA_inversion_unweighted.json`

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
