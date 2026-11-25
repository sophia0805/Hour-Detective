import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from predict import HERE, MODEL_PATH, CAT_COLS, prepare

PARAMS = dict(
    iterations=2130,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    rsm=0.5,
    random_seed=42,
    verbose=0,
)

df = pd.read_csv(HERE / "train.csv")
y_log = np.log1p(df["hours_spent"])
X = prepare(df.drop(columns=["hours_spent", "log_hours", "hours_bucket"]))

model = CatBoostRegressor(cat_features=CAT_COLS, **PARAMS)
model.fit(X, y_log)
model.save_model(str(MODEL_PATH))
print(f"saved {MODEL_PATH} with {len(model.feature_names_)} features")