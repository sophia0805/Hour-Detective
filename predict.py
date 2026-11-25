import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor