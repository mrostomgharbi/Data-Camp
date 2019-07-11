import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import warnings
warnings.filterwarnings('ignore')

problem_title = 'AXA_Policy_Prediction'
_target_column_name = 'Total_Price'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.AirPassengers()

score_types = [
    rw.score_types.NormalizedRMSE(name='normalized_rmse', precision=3),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    
    # Drop duplicated values :#################################################################
    data = data.drop_duplicates()
    ###########################################################################################
    
    y_array = data[_target_column_name]
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
