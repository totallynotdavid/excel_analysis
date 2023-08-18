"""
Autor: David Duran
Fecha: 05/08/2023

Este paquete se utiliza para comprobar si cierta acción va a subir o bajar usando machine learning.
"""

import os
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn import svm, metrics, model_selection
from sklearn.neural_network import MLPRegressor

from excel_analysis.constants import EXCEL_FILE_NAME, COLUMN_NAMES, INDEX_COLUMN, SheetResult
from excel_analysis.helpers import check_data_size, check_data_shape, check_null_values, check_top_5_price_counts, get_column_names, get_head

# Sistema de logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Argumentos
def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta el programa de análisis de acciones")
    parser.add_argument("--debug", help="Mostrar más detalles del proceso",
                        type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()

# Preprocesamiento
def load_data(file_name=EXCEL_FILE_NAME, single_sheet=False): # single_sheet = False para modo de producción
    """
    Cargar los datos de un archivo Excel con múltiples hojas
    """
    try:
        data = pd.read_excel(file_name, sheet_name=None, engine='openpyxl', index_col=INDEX_COLUMN)

        if not data:
            logging.error(f"El archivo '{file_name}' está vacío o no contiene datos validos.")
            return None
        
        first_sheet_key = list(data.keys())[0]
        if single_sheet:
            return {first_sheet_key: data[first_sheet_key]}
        return data

    except FileNotFoundError:
        logging.error(f"El archivo '{file_name}' no fue encontrado. Por favor comprueba la ubicación (path) y el nombre del archivo.")
        return None
    except Exception as e:
        logging.error(f"No se puede abrir el archivo '{file_name}'. Error: {str(e)}")
        return None

def ensure_float64(df):
    """
    Prueba de que todas las columnas son de tipo float64
    """
    if not all(df.dtypes == 'float64'):
        raise ValueError("Todas las columnas deben ser de tipo float64")

def handle_non_numeric_values(df, columns_to_check):
    for column in columns_to_check:
        is_non_numeric = pd.to_numeric(df[column], errors='coerce').isna()

        if is_non_numeric.sum() > 0:
            logging.warning(f"Valores no numéricos encontrados en la columna '{column}'.")
            non_numeric_rows = df[is_non_numeric][column]
            for idx, value in non_numeric_rows.items():
                logging.info(f"Fila: {idx}, Valor: {value}")
            logging.info(f"No te preocupes, los valores no numéricos serán convertidos a NaN.")

        # Convertir los valores no numéricos a NaN usando coerce
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].astype('float64')

    # Reemplazar los NaN con el valor anterior
    df.fillna(method='ffill', inplace=True)
    df['PX_VOLUME'] = df['PX_VOLUME'].astype('float64')

# Procesamiento
def normalize_data(df):
    columns_to_normalize = [COLUMN_NAMES["price"]] + COLUMN_NAMES["features"]
    for column in columns_to_normalize:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

def dividir_datos_entrenamiento_prueba(df):
    """
    Dividir el dataframe en training y test data
    """

    train_size = int(0.8 * len(df))
    datos_entrenamiento = df.iloc[:train_size]
    datos_prueba = df.iloc[train_size:]

    feature_cols = COLUMN_NAMES["features"]
    X_train = datos_entrenamiento[feature_cols].values
    Y_train = datos_entrenamiento[COLUMN_NAMES["detail"]].values.astype('float')

    X_test = datos_prueba[feature_cols].values
    Y_test = datos_prueba[COLUMN_NAMES["detail"]].values.astype('float')

    return X_train, Y_train, X_test, Y_test

def entrenar_regresor_mlp(X_train, Y_train):
    """
    Entrenar un modelo de red neuronal
    """
    modelo_red_neuronal = MLPRegressor(activation='logistic', hidden_layer_sizes=(200), max_iter=1000, solver='adam')
    modelo_red_neuronal.fit(X_train, Y_train)
    return modelo_red_neuronal

def get_optimal_threshold(Y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred) # fpr = Tasa de falsos positivos que salen positivos, tpr = Tasa de positivos que salen positivos
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1-fpr, index=i), 'tf': pd.Series(tpr - (1-fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    optimal_threshold = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds'].values[0]
    return optimal_threshold

def process_stock_data(df, sheet_name, results_list):
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
    df = df[pd.to_numeric(df[COLUMN_NAMES["detail"]], errors='coerce').notnull()]
    df.loc[:, COLUMN_NAMES["detail"]] = df[COLUMN_NAMES["detail"]].astype('float')
    ensure_float64(df)

    X_train, Y_train, X_test, Y_test = dividir_datos_entrenamiento_prueba(df)

    # Modelo de red neuronal
    modelo_red_neuronal = entrenar_regresor_mlp(X_train, Y_train)
    y_pred = modelo_red_neuronal.predict(X_test)

    # Obtener el umbral (threshold) óptimo
    optimal_threshold = get_optimal_threshold(Y_test, y_pred)

    # Sumar todos los valores de Y_test
    conteo_positivos_reales  = np.sum(Y_test)
    predicciones = (y_pred > optimal_threshold).astype(int)
    conteo_positivos_predichos = np.sum(predicciones)

    final_value = conteo_positivos_reales - conteo_positivos_predichos
    results_list.append(SheetResult(sheet_name, final_value))
    logging.info(f"💰 Valor de final de esta hoja: {final_value}")

def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.ERROR)

    test_mode = os.environ.get('TEST_MODE', 'False') == 'True'
    all_data = load_data(single_sheet=False)

    if all_data is None:
        return

    results_list = []

    if test_mode:
        # Una sola hoja para modo de prueba
        logging.info("Ejecutando en modo de prueba")
        df = all_data[list(all_data.keys())[0]]
        process_stock_data(df, "Test Sheet", results_list)
    else:
        # Múltiples hojas para modo de producción
        number_of_sheets = len(all_data)
        logging.info(f"📂 Encontramos {number_of_sheets} hojas en el archivo Excel\n")
        for sheet_name, df in all_data.items():
            process_stock_data(df, sheet_name, results_list)

    # Ordenando los resultados
    sorted_results = sorted(results_list, key=lambda x: x.final_value, reverse=True)

    print("Las 10 mejores 📈:")
    for result in sorted_results[:10]:
        print(f"* Hoja: {result.sheet_name}, Valor: {result.final_value}")

    if number_of_sheets >= 20:
      print("\nLas 10 peores 📉:")
      for result in sorted_results[-10:]:
          print(f"* Hoja: {result.sheet_name}, Valor: {result.final_value}")

if __name__ == "__main__":
    main()
