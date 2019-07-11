import pandas as pd
from sklearn.model_selection import train_test_split


data_path = "../"
############### Customer One-Hot encoded data
df_customers = pd.read_csv(data_path + "customer_info.csv", sep=",", parse_dates=True)
df_customers = df_customers.drop(columns=["time_stamp", "customer_social_group"])
df_customers = pd.get_dummies(df_customers, columns=["customer_origin", "customer_gender", "customer_marital_cd"], drop_first=True)
df_customers = df_customers.drop(columns=["customer_gender_F", "customer_marital_cd_3"])
df_customers[:15]

############### Agents One-Hot encoded data
df_agents = pd.read_csv(data_path + "agent_info.csv", sep=",", parse_dates=True)
df_agents = df_agents.drop(columns="time_stamp")
df_agents = pd.get_dummies(df_agents, columns=["agent_status"], drop_first=True)
df_agents[:15]

############### Policies One-Hot encoded data
df_policy_data = pd.read_csv(data_path + "policy_info.csv", sep=",", parse_dates=True)
df_policy_data = df_policy_data.drop(columns=["policy_status", "last_payment_dt","payment_method", "policy_issue_dt", "time_stamp"])
df_policy_data = pd.get_dummies(df_policy_data, columns=["contact_channel", "policy_status_cd", "product_code"], drop_first=True)
df_policy_data = df_policy_data.drop(columns=["policy_status_cd_B", "product_code_BP_00479053"])
df_policy_data[:15]

############### Joining on customer_id and agent_id
df_policy_allInfo = pd.merge(df_policy_data, df_agents, on='agent_id')
df_policy_allInfo = pd.merge(df_policy_allInfo, df_customers, on='customer_id')
df_policy_allInfo = df_policy_allInfo.drop(columns=["customer_id", "agent_id"]) # we remove the information which is specific to each individual
df_policy_allInfo[:15]

############### Gathering latest policy info
df_policy_latest_info = pd.read_csv(data_path + "latest_policy_info.csv", sep=",", parse_dates=True)
df_policy_latest_info["time_stamp"] = pd.to_datetime(df_policy_latest_info["time_stamp"], format='%Y-%m-%d')
df_policy_latest_info["last_termination_dt"] = pd.to_datetime(df_policy_latest_info["last_termination_dt"], format='%Y-%m-%d')

############### Making target predictions
df_policy_cancelled = df_policy_latest_info.dropna(subset=["policy_id", "last_policy_status"]) #drop Na only for this columns
df_policy_cancelled["status"]= df_policy_latest_info["last_policy_status"].map(lambda x: 1 if x=="active" else 0.) # cancelled = 0.
df_policy_cancelled = df_policy_cancelled.drop(columns=["last_policy_status", "last_termination_dt", "time_stamp"])
df_policy_cancelled[:10]

X = pd.merge(df_policy_allInfo, df_policy_cancelled, on='policy_id')
X = X.dropna() # Apparently there were some hidden NaN values
X = X.drop(columns=["policy_id"])
X_train, X_test = train_test_split(X, test_size=.25, shuffle=True, random_state=42)
X_train.to_csv("data/train.csv", index=False)
X_test.to_csv("data/test.csv", index=False)