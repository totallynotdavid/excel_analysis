"""
Autor: David Duran
Fecha: 05/08/2023

Este paquete se utiliza para comprobar si cierta acción va a subir o bajar usando machine learning.
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn import svm, metrics
from sklearn.neural_network import MLPRegressor

from .constants import EXCEL_FILE_NAME, COLUMN_NAMES
from .helpers import check_data_size, check_data_shape, check_null_values, check_top_5_price_counts, get_column_names, get_head

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Preprocesamiento
def load_data(file_name=EXCEL_FILE_NAME, single_sheet=False): # single_sheet = False para modo de producción
    """
    Cargar los datos de un archivo Excel con múltiples hojas
    """
    try:
        xls = pd.ExcelFile(file_name, engine='openpyxl')
    except FileNotFoundError:
        print(f"🚨 [ERROR]: El archivo '{file_name}' no fue encontrado. Por favor comprueba la ubicación (path) y el nombre del archivo.")
        return None
    except Exception as e:
        print(f"🚨 [ERROR]: No se puede abrir el archivo '{file_name}'. Error: {str(e)}")
        return None

    if single_sheet:
        return pd.read_excel(xls, xls.sheet_names[0], index_col='FECHA')
    else:
        all_sheets = {}
        for sheet_name in xls.sheet_names:
            sheet_data = pd.read_excel(xls, sheet_name, index_col='FECHA')
            if not sheet_data.empty:  # Ignorar hojas vacías
                all_sheets[sheet_name] = sheet_data
        return all_sheets

def ensure_float64(df):
    """
    Prueba de que todas las columnas son de tipo float64
    """
    if not all(df.dtypes == 'float64'):
        raise ValueError("Todas las columnas deben ser de tipo float64")

def handle_non_numeric_values(df, columns_to_check):
    for column in columns_to_check:
        non_numeric = df[pd.to_numeric(df[column], errors='coerce').isna()]
        if not non_numeric.empty:
            logging.warning(f"Valores no numéricos encontrados en la columna '{column}'.")
            for index, row in non_numeric.iterrows():
                logging.info(f"Fila: {index}, Valor: {row[column]}")
            logging.info(f"No te preocupes, los valores no numéricos serán convertidos a NaN.")
        # Convertir los valores no numéricos a NaN usando coerce
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # Reemplazar los NaN con el valor anterior
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
    df.fillna(method='ffill', inplace=True)

# Procesamiento
def normalize_column(df, column_name):
    """
    Normalizar datos en una columna específica del dataframe
    """
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())

def normalize_data(df):
    """
    Normalizar datos en columnas específicas del dataframe
    """
    columns_to_normalize = [COLUMN_NAMES["price"]] + COLUMN_NAMES["features"]
    for column in columns_to_normalize:
        normalize_column(df, column)

def get_training_and_test_data(df):
    """
    Dividir el dataframe en training y test data
    """
    division1_df = df.iloc[0:2886, 0:9]
    division2_df = df.iloc[2887:3607, 0:9]

    feature_cols = COLUMN_NAMES["features"]
    
    X_train = division1_df[feature_cols].values
    Y_train = division1_df['Detalle'].values.astype('float')
    
    X_test = division2_df[feature_cols].values
    Y_test = division2_df['Detalle'].values.astype('float')

    ensure_float64(df)

    return X_train, Y_train, X_test, Y_test

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
    show_logs = os.environ.get('SHOW_LOGS', 'True') == 'True'
    if not show_logs:
        logging.getLogger().setLevel(logging.ERROR)

    test_mode = os.environ.get('TEST_MODE', 'False') == 'True'
    all_data = load_data(single_sheet=False)

    if all_data is None:
        return

    if test_mode:
        # Una sola hoja para modo de prueba
        logging.info("Ejecutando en modo de prueba")
        df = all_data[list(all_data.keys())[0]]
        process_stock_data(df, "Test Sheet")
    else:
        # Múltiples hojas para modo de producción
        number_of_sheets = len(all_data)
        logging.info(f"📂 Encontramos {number_of_sheets} hojas en el archivo Excel\n")
        for sheet_name, df in all_data.items():
            process_stock_data(df, sheet_name)

def process_stock_data(df, sheet_name):
    # Revisar que el dataframe tenga todas las columnas esperadas (price, detail y features)
    expected_columns = [COLUMN_NAMES["price"], COLUMN_NAMES["detail"]] + COLUMN_NAMES["features"]
    if not all(column in df.columns for column in expected_columns):
        logging.error(f"La hoja '{sheet_name}' no contiene todas las columnas esperadas. Ignorando esta hoja.")
        logging.info(f"--------------------------------")
        return

    logging.info(f"Trabajando en el stock: {sheet_name}")

    # Validación de datos en el dataframe
    columns_to_check = [COLUMN_NAMES["price"], COLUMN_NAMES["detail"]] + COLUMN_NAMES["features"]
    handle_non_numeric_values(df, columns_to_check)

    normalize_data(df)
    df = df[pd.to_numeric(df['Detalle'], errors='coerce').notnull()]
    df.loc[:, 'Detalle'] = df['Detalle'].astype('float')

    X_train, Y_train, X_test, Y_test = get_training_and_test_data(df)

    # Modelo de red neuronal
    nn = train_neural_network(X_train, Y_train)
    y_pred = nn.predict(X_test)

    # Obtener el umbral (threshold) óptimo
    optimal_threshold = get_optimal_threshold(Y_test, y_pred)

    # Sumar todos los valores de Y_test
    A = np.sum(Y_test)
    df_temp = pd.DataFrame({'Predicted': y_pred})
    df_temp['threshold_comparison'] = (df_temp['Predicted'] > optimal_threshold).astype(int)
    B = df_temp['threshold_comparison'].sum()

    final_value = A - B

    logging.info(f"💰 Valor de final de esta hoja: {final_value}")

if __name__ == "__main__":
    main()
