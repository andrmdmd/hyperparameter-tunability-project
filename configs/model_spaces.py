from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_SPACES = {
    "xgboost": {
        "model": XGBClassifier,
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [3, 5, 7, 10],
            "model__learning_rate": [0.01, 0.1, 0.3],
            "model__subsample": [0.7, 1.0],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
    },
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "model__C": [0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
        },
    },
}
