import numpy as np
import pandas as pd
import pydotplus
from IPython.display import HTML, display
import tabulate

df_latest_policy_info = pd.read_csv("latest_policy_info.csv", sep=",", parse_dates=True)
df_policy_info = pd.read_csv("policy_info.csv", sep=",", parse_dates=True)
df_customer_info = pd.read_csv("customer_info.csv", sep=",", parse_dates=True)
df_agent_info = pd.read_csv("agent_info.csv", sep=",", parse_dates=True)

from sklearn.tree import DecisionTreeRegressor
def predict_income(data, drop_col = []): #['last_payment_dt', 'policy_issue_dt']):
    pd.options.mode.chained_assignment = None  # default='warn'
    try:
        data.drop(columns=drop_col, inplace=True)
    except:
        print("The columns to drop don't exist in the dataframe, the prediction has failed")
        return


    df_train_income = data.copy()
    df_train_income.dropna(inplace=True)
    y = df_train_income.customer_income
    
    #define train and test data
    df_train_income.drop(columns=['customer_income'], inplace=True)
    df_test_income = data[data['customer_income'].isnull()]
    df_test_income.drop(columns=['customer_income'], inplace=True)
    
    #train the model
    model = DecisionTreeRegressor()
    model.fit(df_train_income, y)
    
    df_test_income['customer_income']= y_predicted
    data.at[df_test_income.index, 'customer_income'] = df_test_income['customer_income']

    print("The prediction failed. The df may have no field customer_income.")

# Processing on df_latest_policy_info
df_latest_policy_info.last_policy_status = df_latest_policy_info.last_policy_status.astype('str')

# Processing on df_policy_info
df_policy_info = df_policy_info.drop('payment_method', axis=1) #unused column

# Processing on df_customer_info -> customer income to find
#predict_income(df_customer_info,drop_col = ["time_stamp"])

# Processing on df_agent_info 


from enum import Enum
class available_db(Enum):
    """Every database available"""
    latest_policy_info = "latest_policy_info"
    policy_info = "policy_info"
    customer_info = "customer_info"
    agent_info = "agent_info"

def single_df(df_name):
    """
    return the requested database
       df_name : available_db.*DATABASE_NAME*
    """
    if df_name == available_db.latest_policy_info:
        return df_latest_policy_info
    if df_name == available_db.policy_info:
        return df_policy_info
    if df_name == available_db.customer_info:
        return df_customer_info
    if df_name == available_db.agent_info:
        return df_agent_info
    else:
        raise ValueError("The specified database name does not exist")


def get_merged(db_to_merge):
    """
    return the merged database from every database put in the list in argument
    
    Use example : get_merged([available_db.policy_info, available_db.latest_policy_info])
    """
    
    merged = df_policy_info.copy() if available_db.policy_info in db_to_merge else df_policy_info[["policy_id","agent_id", "customer_id"]].copy()
    
    for db_name in db_to_merge:
        if (db_name == available_db.policy_info):
            continue #already added
        if (db_name not in available_db): 
            print("Warning : one of the specified database was not found in our available databases")
            continue
        
        if (db_name == available_db.latest_policy_info):
            join_on = "policy_id"
        if (db_name == available_db.agent_info):
            join_on = "agent_id"
        if (db_name == available_db.customer_info):
            join_on = "customer_id"
        merged = pd.merge(merged, single_df(db_name), on=join_on)
    return merged

def allsublist(size,mylist):
    import itertools
    def contains_sublist(lst, sublst):
        n = len(sublst)
        return any((sublst == lst[i:i+n]) for i in xrange(len(lst)-n+1))
    return [i for i in itertools.combinations(mylist,size)]
        