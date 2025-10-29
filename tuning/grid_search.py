from sklearn.model_selection import GridSearchCV


def run_grid_search(model_cls, param_space, X_train, y_train, metric):
    search = GridSearchCV(model_cls(), param_space, scoring=metric, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_
