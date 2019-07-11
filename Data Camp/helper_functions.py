
import numpy as np
import pandas as pd
import pydotplus
from IPython.display import HTML, display
import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')



def create_df_for_agent_analysis(data):
    """
    Creates a new df containing average values of all important variables for each agent

    :param data: pandas dataframe
    """

    mean_per_agent = (data.groupby(['agent_id']).mean())
    median_customer_income_per_agent = data.groupby(['agent_id']).median()['customer_income']

    X = pd.DataFrame(data.agent_id)
    X = X.drop_duplicates()

    X = X.join(mean_per_agent, on='agent_id', how='left')
    X = X.join(median_customer_income_per_agent, on='agent_id', how='left', rsuffix='_median')

    # Replace NaN customer income by the median
    X['customer_income'] = X['customer_income'].fillna(value=X['customer_income'].median())

    # Get the predicted variable- whether the agent performs better than the median agent
    X['total_portfolio'] = X['agent_active_portfolio'] + X['agent_inactive_portfolio']
    X['percent_of_active_portfolio'] = X['agent_active_portfolio'] / X['total_portfolio']
    X['percent_of_portfolio'] = X['total_portfolio'] / X['total_portfolio'].sum()

    X['Score'] = (X['percent_of_active_portfolio'] + X['percent_of_portfolio'])/2.
    X['Better'] = (X['Score'] > (X.median()['Score'])).astype(int)

    return X


def plot_dist_with_mean(data, var_to_plot):
    """
    Plots the distribution of a variable with its mean, median and mode.

    :param data: pd dataframe
    :param var_to_plot: list of str
    :return: plot
    """

    plt.figure(figsize=(18, 12))

    for i, col in enumerate(var_to_plot):
        plt.subplot(2, 2, i + 1)

        mean = data[col].mean()
        median = data[col].median()
        mode = data[col].mode().get_values()[0]

        ax_hist = sns.distplot(data[col], hist=True)
        ax_hist.axvline(mean, color='r', linestyle='--')
        ax_hist.axvline(median, color='g', linestyle='-')
        ax_hist.axvline(mode, color='b', linestyle='-')

        plt.legend({'Mean': mean, 'Median': median, 'Mode': mode})

        plt.xticks(rotation=45)
        plt.title('Distribution of:  ' + col)

    return plt.figure(figsize=(7, 5))




def plot_distributions(data, cols_to_plot):
    """Plots all column distributions given in the list 'cols_to_plot' grouped by agent performance (whether the agent performs better or worse than the average agent).

    :param data:
    :param cols_to_plot:
    :return:
    """

    import warnings
    warnings.filterwarnings('ignore')

    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 30))

    for i, col in enumerate(cols_to_plot):
        plt.subplot(6, 3, i + 1)
        plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=0.4, hspace=0.8)

        if col in ['customer_income', 'total_portfolio']:
            ax_tot = sns.distplot(np.log(data[col]), hist=False, label='Total', color='b')
            sns.distplot(np.log(data[data.Better == 1][col]), hist=False, label='Better than median',
                         color='r')
            sns.distplot(np.log(data[data.Better == 0][col]), hist=False, label='Worse than median',
                         color='black')
            plt.xticks(rotation=45)
            plt.title('Distribution of log ' + col + ' \n according to agent \n better (1) performance')
            plt.legend();
        else:
            sns.distplot(data[col], hist=False, label='Total', color='b')
            sns.distplot(data[data.Better == 1][col], hist=False, label='Better than median', color='r')
            sns.distplot(data[data.Better == 0][col], hist=False, label='Worse than median', color='black')
            plt.xticks(rotation=45)
            plt.title('Distribution of ' + col + ' \n according to agent \n  performance')
            plt.legend();


def plot_mean_agent_variables(data, cols_to_plot):
    """
    Plots all column mean values grouped by agent performance (whether the agent performs better
    or worse than the average agent) in the cols_to_plot list.

    :param data: pandas dataframe
    :param cols_to_plot: list containing the columns to plot
    """

    plt.figure(figsize=(20, 30))

    for i, col in enumerate(cols_to_plot):
        plt.subplot(6, 3, i + 1)
        plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=0.4, hspace=0.8)
        group = data[[col,'Better']].groupby(['Better']).mean()
        sns.barplot(x= group.index, y = group[col], palette=sns.color_palette("Blues_d"))
        plt.xticks(rotation=45)
        plt.title('Average: '+ col + ' \n for agents who perform \n better (1) or worse (0) \n than the median agent')




def select_best_var(data, y, num):
    importance = pd.DataFrame()
    importance['name'] = data.columns

    model = RandomForestClassifier(n_estimators=100)

    model.fit(data, y)
    importance['importance'] = model.feature_importances_

    (pd.Series(model.feature_importances_, index=data.columns).nlargest(num).plot(kind='barh',
                                                                                  title='Feature importance of top features'))

    best_var = importance.sort_values(by=['importance'], ascending=False, inplace=False)
    best_var = best_var['name'][:num]
    best_var = best_var.values

    return best_var