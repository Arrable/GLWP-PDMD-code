import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score

# ================= 1. Global Configuration =================

# --- Core file paths (Please update these to your actual paths) ---
base_dir = r"C:\path\to\your\data"
file_name = "midday_data.xlsx"
file_path = os.path.join(base_dir, file_name)

# Output file paths
model_save_path_paper = os.path.join(base_dir, "xgb_model_seed42_paper_unweighted.json")
model_save_path_full = os.path.join(base_dir, "xgb_model_FULL_DATA_inversion_unweighted.json")

# --- Key column configurations ---
target_col = "md"          # Target variable (Y)
site_col = "Station_ID"    # Site/Station column (used for grouping)
date_col = "date"          # Date column (used for grouping)

# Feature list (X)
feature_cols_manual = [
    "T", "NDVI", "VPD",
    "SM1", "ST1",
    "SM2", "ST2",
    "Sand", "Silt", "Clay"
]

# ================= 2. Data Loading & Spatiotemporal Grouping =================

print(f"📂 Reading file: {file_path}")
if not os.path.exists(file_path):
    print(f"❌ Error: File not found at {file_path}")
    exit()

df = pd.read_excel(file_path)

required_cols = feature_cols_manual + [target_col, site_col, date_col]
df = df.dropna(subset=required_cols)

# Create Group_ID (combining site and date)
df['Group_ID'] = df[site_col].astype(str) + "_" + df[date_col].astype(str)

print("✅ Data preprocessing completed.")
print(f"   Total rows: {len(df)}")
print(f"   Unique Spatiotemporal blocks (Site-Days): {df['Group_ID'].nunique()}")

X = df[feature_cols_manual]
y = df[target_col]
groups = df['Group_ID']

# ================= 3. Stage 1: Optuna Hyperparameter Tuning =================

print("-" * 30)
print(f"🔍 Stage 1: Optuna Hyperparameter Tuning (Unweighted Standard Mode)...")

# Precisely replicate the 20% hold-out test set
gss_tuning = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss_tuning.split(X, y, groups=groups))

X_train_tune = X.iloc[train_idx]
y_train_tune = y.iloc[train_idx]
groups_train_tune = groups.iloc[train_idx]

def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }

    n_groups_inner = groups_train_tune.nunique()
    k_folds = min(5, n_groups_inner)
    gkf = GroupKFold(n_splits=k_folds)

    cv_scores = []

    for t_idx, v_idx in gkf.split(X_train_tune, y_train_tune, groups=groups_train_tune):
        X_tr, X_val = X_train_tune.iloc[t_idx], X_train_tune.iloc[v_idx]
        y_tr, y_val = y_train_tune.iloc[t_idx], y_train_tune.iloc[v_idx]

        model = xgb.XGBRegressor(**param)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        cv_scores.append(score)

    return np.mean(cv_scores)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"🏆 Best Parameters R²: {study.best_value:.4f}")
final_params = study.best_params.copy()
final_params.update({'objective': 'reg:squarederror', 'n_jobs': -1, 'verbosity': 0})

# ================= 4. Stage 2: Stability Verification =================

print("-" * 30)
print("🔄 Stage 2: Strict Stability Check on Train Set (Cross-Validation)...")

n_groups_stab = groups_train_tune.nunique()
k_folds_stab = min(5, n_groups_stab)
gkf_stab = GroupKFold(n_splits=k_folds_stab)

r2_list = []
rmse_list = []
nrmse_list = []

fold = 1
for t_idx, v_idx in gkf_stab.split(X_train_tune, y_train_tune, groups=groups_train_tune):
    X_tr, X_val = X_train_tune.iloc[t_idx], X_train_tune.iloc[v_idx]
    y_tr, y_val = y_train_tune.iloc[t_idx], y_train_tune.iloc[v_idx]

    model = xgb.XGBRegressor(**final_params, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred_val = model.predict(X_val)

    r2 = r2_score(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    data_range = y_val.max() - y_val.min()
    nrmse = rmse / data_range if data_range != 0 else 0

    r2_list.append(r2)
    rmse_list.append(rmse)
    nrmse_list.append(nrmse)

    print(f"   -> Fold {fold}: R²={r2:.4f}, RMSE={rmse:.4f}")
    fold += 1

print("-" * 30)
print("📊 [Paper Metrics: Train Set CV Mean ± Std]")
print(f"   R²    = {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"   RMSE  = {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")

# ================= 5. Stage 3: Final Test Set Evaluation =================

print("-" * 30)
print("🎯 Stage 3: Final Hold-out Test Set Evaluation...")

# Directly use the pure test set isolated in Stage 1
X_test_final = X.iloc[test_idx]
y_test_final = y.iloc[test_idx]

# Train the final evaluation model using the entire 80% training set
paper_model = xgb.XGBRegressor(**final_params, random_state=42)
paper_model.fit(X_train_tune, y_train_tune)
paper_model.save_model(model_save_path_paper)

# Perform ultimate predictions on the 20% hold-out test set
y_pred_final = paper_model.predict(X_test_final)

r2_test = r2_score(y_test_final, y_pred_final)
rmse_test = np.sqrt(mean_squared_error(y_test_final, y_pred_final))

data_range_test = y_test_final.max() - y_test_final.min()
nrmse_test = rmse_test / data_range_test if data_range_test != 0 else 0

print(f"🎯 [Hold-out Test Set Final Performance]")
print(f"   R²    = {r2_test:.4f}")
print(f"   RMSE  = {rmse_test:.4f}")
print(f"   NRMSE = {nrmse_test * 100:.1f}%")

# ================= 6. Stage 4: Full Data Inversion Model Training =================

print("-" * 30)
print("🌍 Stage 4: Training Specialized Inversion Model (100% Data)...")

final_inversion_params = final_params.copy()
final_inversion_params['random_state'] = 42

full_model = xgb.XGBRegressor(**final_inversion_params)
full_model.fit(X, y)

full_model.save_model(model_save_path_full)
print(f"✅ Full data inversion model successfully saved to: {model_save_path_full}")
print("-" * 30)
print("🚀 All tasks completed successfully!")