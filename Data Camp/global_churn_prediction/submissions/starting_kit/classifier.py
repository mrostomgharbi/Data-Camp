from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class Classifier(BaseEstimator):
    
    def __init__(self, n_estimators=40, max_depth=20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, class_weight="balanced", max_depth=self.max_depth)
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        
    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)