from scipy import constants
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.tree import DecisionTreeRegressor

def _encode(X_df):
    merged_data = X_df.copy()
    
    # Drop columns:
    merged_data = merged_data.drop(columns=['policy_id', 'customer_id', 'agent_id','policy_status','time_stamp_x','time_stamp_y', 'product_code'])

    # Transform to time stamp the date columns:
    merged_data.last_payment_dt = pd.to_datetime(merged_data.last_payment_dt)
    merged_data.policy_issue_dt = pd.to_datetime(merged_data.policy_issue_dt)
    
    # Drop unusefull columns : 
    merged_data.drop(columns=['last_payment_dt','payment_method','policy_issue_dt'], inplace=True)
    
    # Transform categorical columns :
    get_dummies_cols = ['nb_riders', 'customer_marital_cd', 'contact_channel', 'policy_status_cd', 'customer_gender', 'customer_origin', 'customer_age' ]
    merged_data = pd.get_dummies(merged_data, columns=get_dummies_cols, drop_first=True)

#     # Drop duplicated values :#################################################################
#     merged_data = merged_data.drop_duplicates()
#     ###########################################################################################
    
#     # Removing outliers in customer_income columns :###########################################
#     merged_data=merged_data[merged_data.customer_income<1e6]
#     ###########################################################################################

    
    return merged_data


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        X_encoded = _encode(X_df)
        self.columns = X_encoded.columns
        return self
        
    def transform(self, X_df):
        merged_data=X_df
        
        merged_data = _encode(merged_data)
        
        # filling nan values for customer_income column : 
        df_train_income = merged_data.copy()
        df_train_income.dropna(inplace=True)
        y = df_train_income.customer_income
        df_train_income.drop(columns=['customer_income'], inplace=True)
        df_test_income = merged_data[merged_data['customer_income'].isnull()]
        df_test_income.drop(columns=['customer_income'], inplace=True)

        model = DecisionTreeRegressor()
        model.fit(df_train_income, y)

        y_predicted = model.predict(df_test_income)
        df_test_income['customer_income']= y_predicted
        merged_data.at[df_test_income.index, 'customer_income'] = df_test_income['customer_income']
        X_empty = pd.DataFrame(columns=self.columns)
        merged_data = pd.concat([X_empty, merged_data], axis=0)
        merged_data = merged_data.fillna(0)
        
        col_to_save=np.intersect1d(merged_data.columns,list(self.columns))
        merged_data = merged_data[col_to_save]

#         # Removing outliers in customer_income columns :###########################################
#         merged_data=merged_data.loc[merged_data.customer_income<1e6]
#         ###########################################################################################
            
#         merged_data = merged_data.values
        
        
        return merged_data


    
