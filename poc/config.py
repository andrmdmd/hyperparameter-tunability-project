from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

OPENML_DATASETS = [
    1590,   # Adult
    1510,   # Breast Cancer Wisconsin (Diagnostic)
    1461,   # Bank Marketing
    24,     # Mushroom
    40945   # Titanic
]
#OPENML_DATASETS = [37, 44]

ML_MODELS = {
    # Decision Tree (rpart in the table)
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "hyperparameters": {
            "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.008},  # cp (complexity parameter for pruning)
            "max_depth": {"type": "int", "low": 12, "high": 27},  # maxdepth
            "min_samples_leaf": {"type": "int", "low": 4, "high": 42},  # minbucket
            "min_samples_split": {"type": "int", "low": 5, "high": 49},  # minsplit
        },
    },
    # Random Forest (ranger in the table)
    "random_forest": {
        "class": RandomForestClassifier,
        "hyperparameters": {
            "n_estimators": {"type": "int", "low": 206, "high": 1740},  # num.trees
            "bootstrap": {"type": "categorical", "choices": [True, False]},  # replace
            "max_features": {"type": "float", "low": 0.323, "high": 0.974},  # sample.fraction
            "min_samples_split": {"type": "float", "low": 0.007, "high": 0.513},  # min.node.size
        },
    },
    # K-Nearest Neighbors
    "knn": {
        "class": KNeighborsClassifier,
        "hyperparameters": {
            "n_neighbors": {"type": "int", "low": 10, "high": 30},  # k
        },
    },
    # # Support Vector Machine
    # "svm": {
    #     "class": SVC,
    #     "hyperparameters": {
    #         "kernel": {"type": "categorical", "choices": ["linear", "rbf", "poly", "sigmoid"]},
    #         "C": {"type": "float", "low": 0.002, "high": 920.582, "log": True},  # cost
    #         "gamma": {"type": "float", "low": 0.003, "high": 18.195, "log": True},
    #         "degree": {"type": "int", "low": 2, "high": 4},
    #     },
    # },
    # # XGBoost
    # "xgboost": {
    #     "class": XGBClassifier,
    #     "hyperparameters": {
    #         "n_estimators": {"type": "int", "low": 921, "high": 4551},  # nrounds
    #         "eta": {"type": "float", "low": 0.002, "high": 0.355},  # eta (learning rate)
    #         "subsample": {"type": "float", "low": 0.545, "high": 0.958},
    #         "booster": {"type": "categorical", "choices": ["gbtree", "gblinear", "dart"]},
    #         "max_depth": {"type": "int", "low": 6, "high": 14},  # max_depth
    #         "min_child_weight": {"type": "float", "low": 1.295, "high": 6.984},
    #         "colsample_bytree": {"type": "float", "low": 0.419, "high": 0.864},
    #         "colsample_bylevel": {"type": "float", "low": 0.335, "high": 0.886},
    #         "lambda": {"type": "float", "low": 0.008, "high": 29.755},  # lambda (L2 reg)
    #         "alpha": {"type": "float", "low": 0.002, "high": 6.105},  # alpha (L1 reg)
    #     },
    # }
}

DEFAULT_SAMPLING_METHODS = ["bayesian", "random", "grid"]

DEFAULT_DATASET_USAGES = [10, 25, 50, 75, 100]

N_TRIALS = 32
