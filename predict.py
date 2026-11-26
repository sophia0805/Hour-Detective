import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
MAX_HOURS = 140.7
META = json.loads((HERE / "blend.json").read_text())

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
    X = X.copy()

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


def prepare(df):
    return add_features(df.drop(columns=["project_id"]))[META["base_columns"]]


def predict(test_path, out_path):
    test = pd.read_csv(test_path)
    Xf = prepare(test)
    w = META["weights"]

    cat = CatBoostRegressor()
    cat.load_model(str(HERE / "model.cbm"))
    p_cat = cat.predict(Xf[cat.feature_names_])

    X_lgb = Xf.copy()
    for col in CAT_COLS:
        X_lgb[col] = X_lgb[col].astype("category")
    booster = lgb.Booster(model_file=str(HERE / "model_lgb.txt"))
    p_lgb = booster.predict(X_lgb)

    X_ohe = pd.get_dummies(Xf, columns=CAT_COLS, dtype=float)
    X_ohe = X_ohe.reindex(columns=META["ohe_columns"], fill_value=0.0).fillna(0).astype(float)

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(HERE / "model_xgb.json"))
    p_xgb = xgb_model.predict(X_ohe)

    r = META["ridge"]
    scaled = (X_ohe.values - np.array(r["mean"])) / np.array(r["scale"])
    p_ridge = scaled @ np.array(r["coef"]) + r["intercept"]

    preds = {"cat": p_cat, "lgb": p_lgb, "xgb": p_xgb, "ridge": p_ridge}
    for name, p in preds.items():
        if not np.isfinite(p).all():
            raise ValueError(f"{name} produced NaN or inf predictions")

    log_pred = sum(w[name] * preds[name] for name in preds)
    hours = np.clip(np.expm1(log_pred), 0, MAX_HOURS)
    pd.DataFrame({"project_id": test["project_id"], "hours_spent": hours}).to_csv(out_path, index=False)


if __name__ == "__main__":
    predict(HERE / "train.csv", HERE / "predictions.csv")