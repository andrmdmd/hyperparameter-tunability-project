from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_SPACES = {
    "xgboost": {
        "model": XGBClassifier,
        "params": {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.7, 1.0],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
    },
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
    },
}
