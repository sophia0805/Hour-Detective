from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.csv"
MODEL_PATH = HERE / "model.cbm"

LABEL_COLS = ["hours_spent", "log_hours", "hours_bucket"]
MAX_HOURS = 140.7

PRIOR_MAP = {"0": 0, "1-2": 1, "3-5": 2, "6+": 3}
AGE_MAP = {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3}
BOOL_MAP = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}

CAT_COLS = ["primary_language", "framework", "hosting_type", "commit_cadence", "project_category"]
BOOL_COLS = [
    "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile", "has_backend",
    "has_database", "has_custom_art", "readme_has_install_instructions", "has_license",
    "has_docs_folder", "url_is_live", "is_mobile_responsive", "has_custom_domain",
    "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial", "is_forked_from_template",
]

PARAMS = dict(
    iterations=2130,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    rsm=0.5,
    random_seed=42,
    verbose=0,
)

def clean(X):
    X = X.copy()
    X["author_prior_projects"] = X["author_prior_projects"].astype(str).map(PRIOR_MAP)
    X["author_account_age_years"] = X["author_account_age_years"].astype(str).map(AGE_MAP)
    for c in BOOL_COLS:
        if c in X.columns:
            X[c] = X[c].map(BOOL_MAP)
    for c in CAT_COLS:
        X[c] = X[c].fillna("missing").astype(str)
    return X

def div(a, b):
    return a / b.replace(0, np.nan)

def add_features(X):
    X = X.copy()
    X["lines_per_active_day"] = div(X.lines_added_total, X.active_commit_days)
    X["loc_per_commit"] = div(X.lines_of_code, X.num_commits)
    X["commits_per_active_day"] = div(X.num_commits, X.active_commit_days)
    X["active_frac"] = div(X.active_commit_days, X.project_span_days + 1)
    X["total_lines_touched"] = np.log1p(X.lines_added_total + X.lines_deleted_total)
    X["biggest_commit_share"] = div(X.max_lines_in_one_commit, X.lines_added_total)
    X["bulk_upload"] = (X.pct_code_in_first_commit > 0.8).astype(float).where(X.pct_code_in_first_commit.notna())
    X["few_commits"] = (X.num_commits <= 3).astype(float).where(X.num_commits.notna())
    X["no_git"] = X.num_commits.isna().astype(float)
    X["n_assets"] = X[["num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files"]].sum(axis=1, min_count=1)
    X["readme_richness"] = X[["readme_num_headings", "readme_num_images", "readme_num_code_blocks"]].sum(axis=1, min_count=1)
    X["loc_x_days"] = np.log1p(X.lines_of_code) * np.log1p(X.active_commit_days)
    X["loc_template_adj"] = np.log1p(X.lines_of_code) * (1 - 0.5 * X.is_forked_from_template.fillna(0))
    return X

def prepare(df):
    X = df.drop(columns=["project_id"] + LABEL_COLS, errors="ignore")
    return add_features(clean(X))

def train():
    df = pd.read_csv(TRAIN_PATH)
    X = prepare(df)
    y_log = np.log1p(df["hours_spent"])
    model = CatBoostRegressor(cat_features=CAT_COLS, **PARAMS)
    model.fit(X, y_log)
    model.save_model(str(MODEL_PATH))
    return model

def load_model():
    if MODEL_PATH.exists():
        model = CatBoostRegressor()
        model.load_model(str(MODEL_PATH))
        return model
    return train()

def predict(test_path, out_path):
    test = pd.read_csv(test_path)
    model = load_model()
    X = prepare(test)[model.feature_names_]
    hours = np.expm1(model.predict(X))
    hours = np.clip(hours, 0, MAX_HOURS)
    out = pd.DataFrame({"project_id": test["project_id"], "hours_spent": hours})
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} predictions to {out_path}")

if __name__ == "__main__":
    predict("test.csv", "predictions.csv")