"""
Autor: David Duran
Fecha: 05/08/2023

Este módulo se utiliza para comprobar si cierta acción va a subir o bajar usando machine learning.
Los pasos que se siguen son:
1. Cargar los datos de un archivo Excel
2. Normalizar los datos
3. Dividir los datos en training y test data
4. Clasificar los datos con SVC (Support Vector Classification)
5. Entrenar un modelo de red neuronal
6. Obtener el umbral (threshold) óptimo
7. Comparar el último valor de y_pred con el umbral óptimo
"""

import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.neural_network import MLPRegressor

# Helper functions (Son utilizados por pytest en test_data_check)
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
def load_data(file_name='Base2.xlsx'):
    """
    Cargar los datos de un archivo Excel con múltiples hojas
    """
    all_sheets = {}
    xls = pd.ExcelFile(file_name, engine='openpyxl')
    for sheet_name in xls.sheet_names:
        all_sheets[sheet_name] = pd.read_excel(xls, sheet_name, index_col='FECHA')
    return all_sheets

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

def get_optimal_threshold(Y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred) # fpr = Tasa de falsos positivos que salen positivos, tpr = Tasa de positivos que salen positivos
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1-fpr, index=i), 'tf': pd.Series(tpr - (1-fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    optimal_threshold = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds'].values[0]
    return optimal_threshold

def main():
    all_data = load_data()
    number_of_sheets = len(all_data)
    print(f"Encontramos {number_of_sheets} hojas en el archivo Excel")

    for sheet_name, df in all_data.items():
        print(f"Trabajando en el stock: {sheet_name}")

        normalize_data(df)
        df = df[pd.to_numeric(df['Detalle'], errors='coerce').notnull()]
        df.loc[:, 'Detalle'] = df['Detalle'].astype('float')

        X_train, Y_train, X_test, Y_test = get_training_and_test_data(df)

        # Modelo SVM
        # clf = train_svm(X_train, Y_train)
        # yhat = clf.predict(X_test)

        # Modelo de red neuronal
        nn = train_neural_network(X_train, Y_train)
        y_pred = nn.predict(X_test)
        # df_temp = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
        # print(df_temp.tail())

        # Obtener el umbral (threshold) óptimo
        optimal_threshold = get_optimal_threshold(Y_test, y_pred)
        # print("Umbral óptimo: " + str(optimal_threshold))

        # Comparar el último valor de y_pred con el umbral óptimo
        last_y_pred = y_pred[-1]
        print(f"Último valor de y_pred: {last_y_pred} y umbral óptimo: {optimal_threshold}")
        if last_y_pred > optimal_threshold:
            print(f"Los precios de {sheet_name} subirán 📈")
        else:
            print(f"Los precios de {sheet_name} bajarán 📉")
        
        print("--------------------------------")

if __name__ == "__main__":
    main()
