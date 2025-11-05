from sklearn.model_selection import GridSearchCV

from tuning.utils import get_model_pipeline


def run_grid_search(model_cls, param_space, X_train, y_train, metric):
    search = GridSearchCV(get_model_pipeline(model_cls), param_space, scoring=metric, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.cv_results_