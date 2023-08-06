"""
This module is used by the pipeline to check if our sources are correct!
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPRegressor

# Helper functions (used by pytest to test the functions in this module)
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
    return df[['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
               'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']].head(number_of_rows)

# Preprocessing
def load_data(file_name='Base1.xlsx'):
    """
    Cargar los datos de un archivo Excel
    """
    return pd.read_excel(file_name, engine='openpyxl', index_col='FECHA')

def ensure_float64(df):
    """
    Prueba de que todas las columnas son de tipo float64
    """
    if not all(df.dtypes == 'float64'):
        raise ValueError("Todas las columnas deben ser de tipo float64")

# Processing
def normalize_column(df, column_name):
    """
    Normalizar datos en una columna específica del dataframe
    """
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())

def normalize_data(df):
    """
    Normalizar datos en columnas específicas del dataframe
    """
    columns_to_normalize = ['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                            'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']
    for column in columns_to_normalize:
        normalize_column(df, column)

def get_training_and_test_data(df):
    """
    Dividir el dataframe en training y test data
    """
    division1_df = df.iloc[0:2886, 0:9]
    division2_df = df.iloc[2887:3607, 0:9]

    feature_cols = ['Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro', 'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']
    
    X_train = division1_df[feature_cols].values
    Y_train = division1_df['Detalle'].values.astype('float')
    
    X_test = division2_df[feature_cols].values
    Y_test = division2_df['Detalle'].values.astype('float')

    ensure_float64(df)

    return X_train, Y_train, X_test, Y_test

def train_svm(X_train, Y_train):
    """
    Clasificar los datos con SVC (Support Vector Classification)
    """
    clf = svm.SVC(kernel='poly', degree=3)
    clf.fit(X_train, Y_train)
    return clf

def train_neural_network(X_train, Y_train):
    """
    Entrenar un modelo de red neuronal
    """
    nn = MLPRegressor(activation='logistic', hidden_layer_sizes=(200), max_iter=1000, solver='adam')
    nn.fit(X_train, Y_train)
    return nn

def main():
    df = load_data()
    normalize_data(df)
    df = df[pd.to_numeric(df['Detalle'], errors='coerce').notnull()]
    df['Detalle'] = df['Detalle'].astype('float')

    X_train, Y_train, X_test, Y_test = get_training_and_test_data(df)

    # SVM model
    clf = train_svm(X_train, Y_train)
    yhat = clf.predict(X_test)

    # Neural Network model
    nn = train_neural_network(X_train, Y_train)
    y_pred = nn.predict(X_test)
    df_temp = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})

if __name__ == "__main__":
    main()
