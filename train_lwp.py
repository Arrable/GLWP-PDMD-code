import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Global Configuration
# ============================================================

# Choose training mode:
# "pd"   = train predawn leaf water potential model
# "md"   = train midday leaf water potential model
# "both" = train both predawn and midday models
MODE = "pd"

# Please update this path to your own data directory
BASE_DIR = r"C:\path\to\your\data"

# Number of Optuna trials
N_TRIALS = 30

# Random seed for reproducibility
RANDOM_SEED = 42

# Grouping columns
SITE_COL = "Station_ID"
DATE_COL = "date"

# Predictor variables
FEATURE_COLS = [
    "T", "NDVI", "VPD",
    "SM1", "ST1",
    "SM2", "ST2",
    "Sand", "Silt", "Clay"
]

# Configuration for predawn and midday models
MODEL_CONFIGS = {
    "pd": {
        "file_name": "predawn_data.xlsx",
        "target_col": "pd",
        "model_prefix": "predawn"
    },
    "md": {
        "file_name": "midday_data.xlsx",
        "target_col": "md",
        "model_prefix": "midday"
    }
}


# ============================================================
# 2. Utility Functions
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate R2, RMSE and NRMSE."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    data_range = y_true.max() - y_true.min()
    nrmse = rmse / data_range if data_range != 0 else np.nan

    return r2, rmse, nrmse


def load_and_prepare_data(file_path, target_col):
    """Load data and create site-date groups."""

    print(f"\n📂 Reading file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path)

    required_cols = FEATURE_COLS + [target_col, SITE_COL, DATE_COL]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()

    df["Group_ID"] = (
        df[SITE_COL].astype(str) + "_" + df[DATE_COL].astype(str)
    )

    print("✅ Data preprocessing completed.")
    print(f"   Total rows: {len(df)}")
    print(f"   Unique site-date groups: {df['Group_ID'].nunique()}")

    if df["Group_ID"].nunique() < 2:
        raise ValueError("At least two unique site-date groups are required.")

    X = df[FEATURE_COLS]
    y = df[target_col]
    groups = df["Group_ID"]

    return df, X, y, groups


def tune_hyperparameters(X_train, y_train, groups_train):
    """Tune XGBoost hyperparameters using Optuna and GroupKFold."""

    print("-" * 50)
    print("🔍 Stage 1: Optuna Hyperparameter Tuning")

    def objective(trial):

        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
            "verbosity": 0
        }

        n_groups = groups_train.nunique()
        n_splits = min(5, n_groups)

        if n_splits < 2:
            raise ValueError("At least two groups are required for GroupKFold.")

        gkf = GroupKFold(n_splits=n_splits)

        cv_scores = []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):

            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)

            cv_scores.append(r2)

        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print(f"🏆 Best CV R²: {study.best_value:.4f}")
    print("🏆 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    final_params = study.best_params.copy()
    final_params.update({
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbosity": 0
    })

    return final_params, study.best_value


def cross_validate_train_set(X_train, y_train, groups_train, final_params):
    """Evaluate model stability using GroupKFold on the training set."""

    print("-" * 50)
    print("🔄 Stage 2: Stability Check on Training Set")

    n_groups = groups_train.nunique()
    n_splits = min(5, n_groups)

    if n_splits < 2:
        raise ValueError("At least two groups are required for GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)

    r2_list = []
    rmse_list = []
    nrmse_list = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_train, y_train, groups=groups_train),
        start=1
    ):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = xgb.XGBRegressor(**final_params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)

        r2, rmse, nrmse = calculate_metrics(y_val, y_pred)

        r2_list.append(r2)
        rmse_list.append(rmse)
        nrmse_list.append(nrmse)

        print(
            f"   Fold {fold}: "
            f"R²={r2:.4f}, RMSE={rmse:.4f}, NRMSE={nrmse * 100:.1f}%"
        )

    cv_metrics = {
        "cv_r2_mean": float(np.mean(r2_list)),
        "cv_r2_std": float(np.std(r2_list)),
        "cv_rmse_mean": float(np.mean(rmse_list)),
        "cv_rmse_std": float(np.std(rmse_list)),
        "cv_nrmse_mean": float(np.nanmean(nrmse_list)),
        "cv_nrmse_std": float(np.nanstd(nrmse_list))
    }

    print("-" * 50)
    print("📊 Training Set CV Mean ± Std")
    print(f"   R²    = {cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}")
    print(f"   RMSE  = {cv_metrics['cv_rmse_mean']:.4f} ± {cv_metrics['cv_rmse_std']:.4f}")
    print(f"   NRMSE = {cv_metrics['cv_nrmse_mean'] * 100:.1f}% ± {cv_metrics['cv_nrmse_std'] * 100:.1f}%")

    return cv_metrics


def evaluate_holdout_test_set(
    X_train,
    y_train,
    X_test,
    y_test,
    final_params,
    model_save_path
):
    """Train model on the 80% training set and evaluate it on the 20% hold-out test set."""

    print("-" * 50)
    print("🎯 Stage 3: Final Hold-out Test Set Evaluation")

    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train, y_train)

    model.save_model(model_save_path)

    y_pred = model.predict(X_test)

    r2, rmse, nrmse = calculate_metrics(y_test, y_pred)

    test_metrics = {
        "test_r2": float(r2),
        "test_rmse": float(rmse),
        "test_nrmse": float(nrmse)
    }

    print("🎯 Hold-out Test Set Performance")
    print(f"   R²    = {r2:.4f}")
    print(f"   RMSE  = {rmse:.4f}")
    print(f"   NRMSE = {nrmse * 100:.1f}%")
    print(f"✅ Paper model saved to: {model_save_path}")

    return model, test_metrics


def train_full_inversion_model(X, y, final_params, model_save_path):
    """Train the final inversion model using 100% of the data."""

    print("-" * 50)
    print("🌍 Stage 4: Training Full-data Inversion Model")

    model = xgb.XGBRegressor(**final_params)
    model.fit(X, y)

    model.save_model(model_save_path)

    print(f"✅ Full-data inversion model saved to: {model_save_path}")

    return model


def train_one_model(model_key):
    """Complete workflow for one target variable."""

    if model_key not in MODEL_CONFIGS:
        raise ValueError("model_key must be 'pd' or 'md'.")

    config = MODEL_CONFIGS[model_key]

    file_path = os.path.join(BASE_DIR, config["file_name"])
    target_col = config["target_col"]
    model_prefix = config["model_prefix"]

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    model_save_path_paper = os.path.join(
        output_dir,
        f"xgb_model_{model_prefix}_seed42_paper_unweighted.json"
    )

    model_save_path_full = os.path.join(
        output_dir,
        f"xgb_model_{model_prefix}_FULL_DATA_inversion_unweighted.json"
    )

    metrics_save_path = os.path.join(
        output_dir,
        f"metrics_{model_prefix}.json"
    )

    params_save_path = os.path.join(
        output_dir,
        f"best_params_{model_prefix}.json"
    )

    print("\n" + "=" * 60)
    print(f"🚀 Training model for: {model_prefix.upper()} leaf water potential")
    print("=" * 60)

    df, X, y, groups = load_and_prepare_data(file_path, target_col)

    # Split data into 80% training and 20% hold-out test set by site-date groups
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print("-" * 50)
    print("📌 Data split completed.")
    print(f"   Training rows: {len(X_train)}")
    print(f"   Test rows: {len(X_test)}")
    print(f"   Training groups: {groups_train.nunique()}")
    print(f"   Test groups: {groups.iloc[test_idx].nunique()}")

    # Stage 1: Hyperparameter tuning
    final_params, best_cv_r2 = tune_hyperparameters(
        X_train,
        y_train,
        groups_train
    )

    # Save best parameters
    with open(params_save_path, "w", encoding="utf-8") as f:
        json.dump(final_params, f, indent=4)

    # Stage 2: Stability check
    cv_metrics = cross_validate_train_set(
        X_train,
        y_train,
        groups_train,
        final_params
    )

    # Stage 3: Final hold-out test evaluation
    paper_model, test_metrics = evaluate_holdout_test_set(
        X_train,
        y_train,
        X_test,
        y_test,
        final_params,
        model_save_path_paper
    )

    # Stage 4: Full-data inversion model training
    full_model = train_full_inversion_model(
        X,
        y,
        final_params,
        model_save_path_full
    )

    # Save all metrics
    all_metrics = {
        "model": model_prefix,
        "target_col": target_col,
        "input_file": config["file_name"],
        "total_rows": int(len(df)),
        "total_groups": int(groups.nunique()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_groups": int(groups_train.nunique()),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "best_optuna_cv_r2": float(best_cv_r2),
        **cv_metrics,
        **test_metrics
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print("-" * 50)
    print(f"📁 Metrics saved to: {metrics_save_path}")
    print(f"📁 Best parameters saved to: {params_save_path}")
    print(f"✅ {model_prefix.upper()} model workflow completed.")

    return all_metrics


# ============================================================
# 3. Main Program
# ============================================================

if __name__ == "__main__":

    if MODE == "both":
        train_one_model("pd")
        train_one_model("md")
    elif MODE in ["pd", "md"]:
        train_one_model(MODE)
    else:
        raise ValueError("MODE must be 'pd', 'md', or 'both'.")

    print("\n🚀 All tasks completed successfully!")