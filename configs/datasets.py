from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# mock datasets for now
def load_dataset(name):
    if name == "iris":
        data = load_iris(as_frame=True)
    elif name == "wine":
        data = load_wine(as_frame=True)
    elif name == "cancer":
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = data["data"], data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    return X_train, X_test, y_train, y_test
