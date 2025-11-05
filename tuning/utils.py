from sklearn.pipeline import Pipeline

from configs.datasets import make_preprocessor

def get_model_pipeline(model_cls):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", model_cls())])