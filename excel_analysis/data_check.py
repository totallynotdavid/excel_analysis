"""
This project is used by the pipeline to check if our sources are correct!
"""

import os
import numpy as np
from datetime import datetime
import pandas as pd
from pandas_datareader import data

def load_data():
    df = pd.read_excel('Base1.xlsx',engine='openpyxl',index_col = 'FECHA')
    return df

def normalize_column(df, column_name):
    """
    Normalize the values of a column between 0 and 1
    Theory: $x^{'}_{i} = \\frac{ x_i - x_{min} }{ x_{max} - x_{min}}$
    """
    df[column_name] = (df[column_name] - df[column_name].min())/(df[column_name].max()-df[column_name].min())

def normalize_data(df):
    columns_to_normalize = ['Precio','Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                            'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']
    for column in columns_to_normalize:
        normalize_column(df, column)

def check_data_size(df):
    return df.size

def check_data_shape(df):
    return df.shape

def check_null_values(df):
    return df.isnull().sum()

def check_top_5_price_counts(df):
    return df['Precio'].value_counts().head(5)

def get_column_names(df):
    return df.columns

def get_head(df, number_of_rows=5):
    return df[['Precio','Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
               'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']].head(number_of_rows)

df = load_data()
normalize_data(df)
print(get_head(df))
