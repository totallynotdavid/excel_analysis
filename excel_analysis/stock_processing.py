"""
Autor: David Duran
Fecha de creación: 05/08/2023
Fecha de moficación: 23/10/2023

Este paquete se utiliza para comprobar si cierta acción va a subir o bajar usando machine learning.
"""

import argparse
import logging
import pandas as pd
import numpy as np

# Importar funciones locales
from excel_analysis.constants import EXCEL_FILE_NAME, COLUMN_NAMES, TRAIN_TEST_SPLIT_RATIO, SheetResult
from excel_analysis.utils.data_loaders import get_valid_sheets, load_data
from excel_analysis.models.neural_networks import entrenar_regresor_mlp, get_optimal_threshold
from excel_analysis.utils.data_preprocessors import ensure_float64, handle_non_numeric_values, normalize_data, dividir_datos_entrenamiento_prueba
from excel_analysis.utils.grading_system import assign_stock_grade

# Configuración del logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def parse_argumentos():
    """
    Parsea y valida los argumentos proporcionados al script.

    Retorna:
        argparse.Namespace: Objeto que contiene los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Analiza hojas de cálculo para predecir el comportamiento de las acciones utilizando machine learning."
    )
    parser.add_argument(
        "--debug",
        help="Activa el modo de depuración para obtener una salida detallada del proceso.",
        type=lambda x: (str(x).lower() == 'true'),
        default=False
    )
    return parser.parse_args()

def process_stock_data(df, sheet_name, results_list):
    # Revisar que el dataframe tenga todas las columnas esperadas (price, detail y features)
    columnas_requeridas = [COLUMN_NAMES["price"], COLUMN_NAMES["detail"]] + COLUMN_NAMES["features"]

    if not set(columnas_requeridas).issubset(df.columns):
        logging.error(f"La hoja '{sheet_name}' no contiene todas las columnas esperadas. Ignorando esta hoja.")
        return

    df.dropna(subset=columnas_requeridas, inplace=True)

    if df.empty or df.isnull().any().any():
        logging.warning(f"La hoja '{sheet_name}' contiene datos inconsistentes o vacíos. Ignorando esta hoja.")
        return

    logging.info(f"Procesando stock: {sheet_name}")

    # Validación de datos en el dataframe
    handle_non_numeric_values(df, columnas_requeridas)
    normalize_data(df)
    df = df[pd.to_numeric(df[COLUMN_NAMES["detail"]], errors='coerce').notnull()]
    df.loc[:, COLUMN_NAMES["detail"]] = df[COLUMN_NAMES["detail"]].astype('float')
    ensure_float64(df, columnas_requeridas)

    X_train, Y_train, X_test, Y_test = dividir_datos_entrenamiento_prueba(df, TRAIN_TEST_SPLIT_RATIO)

    if len(X_train) == 0 or len(Y_train) == 0:
        logging.warning(f"La hoja '{sheet_name}' no tiene suficientes datos para entrenar el modelo. Ignorando esta hoja.")
        return

    # Modelo de red neuronal
    modelo_red_neuronal = entrenar_regresor_mlp(X_train, Y_train)
    y_pred = modelo_red_neuronal.predict(X_test)

    # Asignar una calificación a la acción
    stock_grade = assign_stock_grade(df, y_pred, Y_test)

    # Obtener el umbral (threshold) óptimo
    optimal_threshold = get_optimal_threshold(Y_test, y_pred)
    conteo_positivos_reales  = np.sum(Y_test)
    conteo_positivos_predichos = np.sum((y_pred > optimal_threshold).astype(int))

    final_value = conteo_positivos_reales - conteo_positivos_predichos
    results_list.append(SheetResult(sheet_name, final_value, stock_grade, optimal_threshold))
    logging.info(f"💰 Valor final de esta hoja: {final_value}, Threshold: {optimal_threshold}, Grado: {stock_grade}")

# Programa principal
def main():
    args = parse_argumentos()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.ERROR)

    valid_sheets = get_valid_sheets(EXCEL_FILE_NAME)
    if not valid_sheets:
        logging.error("No se encontraron hojas válidas en el archivo Excel.")
        return

    logging.info(f"📂 Encontramos {len(valid_sheets)} hojas válidas en el archivo Excel\n")

    all_data = load_data(sheets_to_load=valid_sheets, single_sheet=False)

    if all_data is None:
        return

    results = []

    for sheet_name in valid_sheets:
        if sheet_name in all_data:
            process_stock_data(all_data[sheet_name], sheet_name, results)

    # Ordenando los resultados
    sorted_results = sorted(results, key=lambda x: x.final_value, reverse=True)
    display_top_stocks(sorted_results, valid_sheets)

def display_top_stocks(sorted_results, valid_sheets):
    """
    Mostar los resultados de las acciones en orden de mejor a peor.
    """
    print("Resumen de las acciones:")

    print("\nLas 10 mejores 📈:")
    for result in sorted_results[:10]:
        print(f"* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f}")

    if len(valid_sheets) >= 20:
        print("\nLas 10 peores 📉:")
        for result in sorted_results[-10:]:
            print(f"* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Sucedió un error: {str(e)}")
