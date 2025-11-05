import numpy as np
import openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def load_dataset(id):
    X, y = fetch_dataset(id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test


def fetch_dataset(did):
    ds = openml.datasets.get_dataset(did)
    print("Fetching dataset:", ds.name)
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    y = LabelEncoder().fit_transform(y)
    return X, y


def make_preprocessor():
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(
        transformers=[
            (
                "numeric_preprocessing",
                num_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical_preprocessing",
                cat_pipeline,
                make_column_selector(dtype_include=['category', 'object']),
            ),
        ],
        remainder="drop",
    )