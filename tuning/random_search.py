from sklearn.model_selection import RandomizedSearchCV


def run_random_search(model_cls, param_space, X_train, y_train, metric):
    search = RandomizedSearchCV(
        model_cls(), param_space, n_iter=20, scoring=metric, cv=3, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_
