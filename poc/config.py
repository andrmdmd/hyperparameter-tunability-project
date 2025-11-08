from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

OPENML_DATASETS = [
    # 1590,   # Adult
    1510,   # Breast Cancer Wisconsin (Diagnostic)
    # 1461,   # Bank Marketing
    # 24,     # Mushroom
    40945   # Titanic
]
#OPENML_DATASETS = [37, 44]

ML_MODELS = {
    "glmnet": {
        "class": LogisticRegression,
        "init_params": {
            "penalty": "elasticnet",
            "solver": "saga",
            "max_iter": 5000,
            "random_state": 42,
        },
        "scale": True,
        "hyperparameters": {
            "l1_ratio": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "grid_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
            "C": {
                "type": "float",
                "low": -10.0,
                "high": 10.0,
                "transform": "pow2",
                "grid_values": [-10.0, -5.0, 0.0, 5.0, 10.0],
            },
        },
    },
    "rpart": {
        "class": DecisionTreeClassifier,
        "init_params": {"random_state": 42},
        "hyperparameters": {
            "ccp_alpha": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "grid_values": [0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
            },
            "max_depth": {
                "type": "int",
                "low": 1,
                "high": 30,
                "grid_values": [1, 3, 5, 10, 20, 30],
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 60,
                "grid_values": [1, 2, 5, 10, 20, 40, 60],
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 60,
                "grid_values": [2, 5, 10, 20, 40, 60],
            },
        },
    },
    "kknn": {
        "class": KNeighborsClassifier,
        "scale": True,
        "hyperparameters": {
            "n_neighbors": {
                "type": "int",
                "low": 1,
                "high": 30,
                "grid_values": [1, 3, 5, 10, 15, 20, 30],
            }
        },
    },
    # "svm": {
    #     "class": SVC,
    #     "scale": True,
    #     "hyperparameters": {
    #         "kernel": {
    #             "type": "categorical",
    #             "choices": ["linear", "rbf", "poly", "sigmoid"],
    #         },
    #         "C": {
    #             "type": "float",
    #             "low": -10.0,
    #             "high": 10.0,
    #             "transform": "pow2",
    #             "grid_values": [-10.0, -5.0, 0.0, 5.0, 10.0],
    #         },
    #         "gamma": {
    #             "type": "float",
    #             "low": -10.0,
    #             "high": 10.0,
    #             "transform": "pow2",
    #             "grid_values": [-10.0, -5.0, 0.0, 5.0, 10.0],
    #         },
    #         "degree": {
    #             "type": "int",
    #             "low": 2,
    #             "high": 5,
    #             "grid_values": [2, 3, 4, 5],
    #         },
    #     },
    # },
    "ranger": {
        "class": RandomForestClassifier,
        "init_params": {"random_state": 42, "n_jobs": -1},
        "hyperparameters": {
            "n_estimators": {
                "type": "int",
                "low": 1,
                "high": 2000,
                "grid_values": [50, 100, 250, 500, 1000, 2000],
            },
            "bootstrap": {
                "type": "categorical",
                "choices": [True, False],
            },
            "max_samples": {
                "type": "float",
                "low": 0.1,
                "high": 1.0,
                "grid_values": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
            "max_features": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "dataset_transform": "max_features",
                "grid_values": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
            "min_samples_leaf": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "dataset_transform": "min_samples_leaf_n_power",
                "dataset_transform_source": "raw",
                "grid_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "init_params": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "n_jobs": -1,
        },
        "hyperparameters": {
            "n_estimators": {
                "type": "int",
                "low": 1,
                "high": 5000,
                "grid_values": [50, 100, 250, 500, 1000, 2000, 3000, 5000],
            },
            "learning_rate": {
                "type": "float",
                "low": -10.0,
                "high": 0.0,
                "transform": "pow2",
                "grid_values": [-10.0, -7.0, -5.0, -3.0, -1.0, 0.0],
            },
            "subsample": {
                "type": "float",
                "low": 0.1,
                "high": 1.0,
                "grid_values": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
            "booster": {
                "type": "categorical",
                "choices": ["gbtree", "dart"],
            },
            "max_depth": {
                "type": "int",
                "low": 1,
                "high": 15,
                "grid_values": [1, 3, 5, 7, 10, 12, 15],
            },
            "min_child_weight": {
                "type": "float",
                "low": 0.0,
                "high": 7.0,
                "transform": "pow2",
                "grid_values": [0.0, 1.0, 3.0, 5.0, 7.0],
            },
            "colsample_bytree": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "grid_values": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
            "colsample_bylevel": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "grid_values": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
            "reg_lambda": {
                "type": "float",
                "low": -10.0,
                "high": 10.0,
                "transform": "pow2",
                "grid_values": [-10.0, -5.0, 0.0, 5.0, 10.0],
            },
            "reg_alpha": {
                "type": "float",
                "low": -10.0,
                "high": 10.0,
                "transform": "pow2",
                "grid_values": [-10.0, -5.0, 0.0, 5.0, 10.0],
            },
        },
    },
}

DEFAULT_SAMPLING_METHODS = ["bayesian", "random", "grid"]

DEFAULT_DATASET_USAGE = 10

OPTIONAL_DATASETS = [1590, 1461, 24, 40945]

N_TRIALS = 7