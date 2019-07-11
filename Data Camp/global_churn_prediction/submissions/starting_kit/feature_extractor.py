import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class FeatureExtractor(object):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_df, y):
        self.scaler.fit(X_df)

    def transform(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        return X_scaled