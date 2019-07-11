import tabulate
from IPython.display import HTML, display
import pandas as pd

# display functions 
def display_data_types(df):
    """
    The functions displays the data types of the dataframe.
    :param df: pandas dataframe whose datatypes are to be displayed
    """
    
    display(HTML(tabulate.tabulate(
    [["<b>Data types"]], tablefmt='html')))
    print(df.dtypes)    

def show_values(df):
    """
    The functions displays the unique values of each column that has an object datatype.
    :param df: pandas dataframe whose unique values are to be displayed
    """
    
    object_cols = [col for col in df.columns if df[col].dtypes == 'object']
    
    for col in object_cols:
        display(HTML(tabulate.tabulate([["<b> Data types of column", col, pd.unique(df[col])]], tablefmt='html')))
        

# Show the percentage of missing values for each column (We create a function because we will use it repeatedly)
def display_na(df): 
    """
    The function displays the percentage of missing values for each column of the dataframe.
    :param df: pandas dataframe whose missing values are to be displayed
    """
    
    display(HTML("<h4>Percentage of missing variables for each feature:"))
    print(df.isnull().sum(axis=0) * 100 / len(df))
    