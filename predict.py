from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "saved_models"

PRIOR_MAP = {"0": 0, "1-2": 1, "3-5": 2, "6+": 3}
AGE_MAP = {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3}
BOOL_MAP = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}

CAT_COLS = [
    "primary_language",
    "framework",
    "hosting_type",
    "commit_cadence",
    "project_category",
]

BOOL_COLS = [
    "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile",
    "has_backend", "has_database", "has_custom_art",
    "readme_has_install_instructions", "has_license", "has_docs_folder",
    "url_is_live", "is_mobile_responsive", "has_custom_domain",
    "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial",
    "is_forked_from_template",
]

TRAIN_COLS = ["project_id", "hours_spent", "log_hours", "hours_bucket"]

def clean_features(X):
    X = X.copy()
    X["author_prior_projects"] = X["author_prior_projects"].astype(str).map(PRIOR_MAP)
    X["author_account_age_years"] = X["author_account_age_years"].astype(str).map(AGE_MAP)

    for column in BOOL_COLS:
        if column in X.columns:
            X[column] = X[column].map(BOOL_MAP)
    for column in CAT_COLS:
        X[column] = X[column].fillna("missing").astype(str)
    return X


def divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def add_features(X):
    X = clean_features(X)

    X["lines_per_active_day"] = divide(X.lines_added_total, X.active_commit_days)
    X["loc_per_commit"] = divide(X.lines_of_code, X.num_commits)
    X["commits_per_active_day"] = divide(X.num_commits, X.active_commit_days)
    X["active_frac"] = divide(X.active_commit_days, X.project_span_days + 1)
    X["total_lines_touched"] = np.log1p(X.lines_added_total + X.lines_deleted_total)
    X["biggest_commit_share"] = divide(X.max_lines_in_one_commit, X.lines_added_total)
    X["n_assets"] = X[[
        "num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files"
    ]].sum(axis=1, min_count=1)
    X["readme_richness"] = X[[
        "readme_num_headings", "readme_num_images", "readme_num_code_blocks"
    ]].sum(axis=1, min_count=1)
    X["loc_x_days"] = np.log1p(X.lines_of_code) * np.log1p(X.active_commit_days)
    X["loc_template_adj"] = (
        np.log1p(X.lines_of_code)
        * (1 - 0.5 * X.is_forked_from_template.fillna(0))
    )
    X["log_code_velocity"] = np.log1p(divide(X["lines_added_total"], X["active_commit_days"]))
    X["loc_per_file"] = np.log1p(divide(X["lines_of_code"], X["num_source_files"]))
    X["density_active_days"] = divide(X["active_commit_days"], X["project_span_days"] + 1)
    X["log_commits_intensity"] = np.log1p(divide(X["num_commits"], X["project_span_days"] + 1))
    X["non_code_file_ratio"] = divide(X["n_assets"], X["num_source_files"] + X["n_assets"] + 1)

    return X

def load_models(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)

    cb_model = CatBoostRegressor()
    cb_model.load_model(str(model_dir / "catboost_model.cbm"))

    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm_model.txt"))

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(model_dir / "xgboost_model.json"))

    scaler = joblib.load(model_dir / "ridge_scaler.pkl")
    ridge = joblib.load(model_dir / "ridge_model.pkl")
    weights = joblib.load(model_dir / "ensemble_weights.pkl")

    return {
        "catboost": cb_model,
        "lightgbm": lgb_model,
        "xgboost": xgb_model,
        "scaler": scaler,
        "ridge": ridge,
        "weights": weights,
    }


def ohe_columns(models):
    scaler = models["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    names = models["xgboost"].get_booster().feature_names
    if names:
        return list(names)
    raise RuntimeError(
        "Could not recover one-hot column order: fit the scaler on a DataFrame "
        "or save X_ohe.columns alongside the models."
    )

def predict(X_raw, models=None):
    if models is None:
        models = load_models()

    X_raw = X_raw.drop(columns=[c for c in TRAIN_COLS if c in X_raw.columns])

    X_cleaned = add_features(X_raw)

    cb_model = models["catboost"]
    X_cb = X_cleaned[cb_model.feature_names_]

    lgb_model = models["lightgbm"]
    X_lgb = X_cleaned.copy()
    for col in CAT_COLS:
        X_lgb[col] = X_lgb[col].astype("category")
    X_lgb = X_lgb[lgb_model.feature_name()]

    X_ohe = pd.get_dummies(X_cleaned, columns=CAT_COLS, drop_first=True, dtype=float)
    X_ohe = (
        X_ohe.reindex(columns=ohe_columns(models), fill_value=0.0)
        .fillna(0)
        .astype(float)
    )

    preds = {
        "catboost": cb_model.predict(X_cb),
        "lightgbm": lgb_model.predict(X_lgb),
        "xgboost": models["xgboost"].predict(X_ohe),
        "ridge": models["ridge"].predict(models["scaler"].transform(X_ohe)),
    }
    for name, p in preds.items():
        if not np.isfinite(p).all():
            raise ValueError(f"{name} produced NaN or inf predictions")

    weights = models["weights"]
    final_log_pred = sum(weights[name] * preds[name] for name in preds)

    return np.clip(np.expm1(final_log_pred), 0, None)


if __name__ == "__main__":
    test_path = HERE / "test.csv"
    out_path = HERE / "predictions.csv"
    
    df = pd.read_csv(test_path)
    models = load_models()
    hours = predict(df, models)
    
    out = pd.DataFrame({"project_id": df["project_id"], "hours_spent": hours})
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions to {out_path}")