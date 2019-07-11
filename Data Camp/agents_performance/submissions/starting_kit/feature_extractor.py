
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.best_var = None
        pass

    def fit(self, X, y):
        self.best_var = select_best_var(X, y, 7)
        return self

    def transform(self, X):
        X = X[self.best_var]
        return X


def select_best_var(data, y, num, plot_=True):
    importance = pd.DataFrame()
    importance['name'] = data.columns

    model = RandomForestClassifier(n_estimators=100)

    model.fit(data, y)
    importance['importance'] = model.feature_importances_

    if plot_:
        (pd.Series(model.feature_importances_, index=data.columns).nlargest(num).plot(kind='barh',
                                                                                      title='Feature importance of top features'))

    best_var = importance.sort_values(by=['importance'], ascending=False, inplace=False)
    best_var = best_var['name'][:num]
    best_var = best_var.values

    return best_var