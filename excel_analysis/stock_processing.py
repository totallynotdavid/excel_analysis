"""
Autor: David Duran
Fecha de creación: 05/08/2023
Fecha de moficación: 03/01/2024

Este paquete se utiliza para comprobar si cierta acción va a subir o bajar usando machine learning.
"""

import argparse
import logging
import numpy as np

# Importar funciones locales
from excel_analysis.constants import EXCEL_FILE_NAME, COLUMN_NAMES, SheetResult
from excel_analysis.utils.data_loaders import get_valid_sheets, load_data
from excel_analysis.models.neural_networks import obtener_threshold_optimo
from excel_analysis.utils.grading_system import (
    assign_stock_grade,
    assign_performance_grade,
    assign_final_value_grade,
)
from excel_analysis.utils.display_results import (
    mostrar_top_stocks,
    mostrar_distribucion_puntaje,
)
from excel_analysis.utils.data_validation import validar_dataframe
from excel_analysis.utils.entrenamiento import entrenar_y_predecir
from excel_analysis.store_data import store_results_to_json, store_results_to_excel

# Configuración del logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def parse_argumentos():
    """
    Parsea y valida los argumentos proporcionados al script.
    """
    parser = argparse.ArgumentParser(
        description="Analiza hojas de cálculo para predecir el comportamiento de las acciones utilizando machine learning."
    )
    parser.add_argument(
        "--debug",
        help="Activa el modo de depuración para obtener una salida detallada del proceso.",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
    )
    return parser.parse_args()


def process_stock_data(df, sheet_name, results_list):
    """
    Procesar los datos de una hoja de cálculo para predecir el comportamiento de una acción.
    Esta función se encarga de entrenar el modelo, predecir los valores y asignar una calificación a la acción.
    Llama a las funciones de otros módulos para realizar estas tareas.

    Parámetros:
    - df: DataFrame que contiene los datos de la acción.
    - sheet_name: Nombre de la hoja de cálculo.
    - results_list: Lista donde se almacenan los resultados de cada acción.
    """
    columnas_requeridas = [
        COLUMN_NAMES["price"],
        COLUMN_NAMES["detail"],
    ] + COLUMN_NAMES["features"]

    if not validar_dataframe(df, columnas_requeridas):
        logging.warning(
            f"La hoja '{sheet_name}' contiene datos inconsistentes o vacíos. Ignorando esta hoja."
        )
        return

    logging.info(f"Procesando stock: {sheet_name}")

    modelo, y_pred, Y_test = entrenar_y_predecir(df, columnas_requeridas)

    if modelo is None:
        logging.warning(
            f"La hoja '{sheet_name}' no tiene suficientes datos para entrenar el modelo. Ignorando esta hoja."
        )
        return

    # Asignar una calificación a la acción
    stock_grade = assign_stock_grade(df, y_pred, Y_test)
    predicted_return = np.sum(y_pred)

    # Obtener el umbral (threshold) óptimo
    optimal_threshold = obtener_threshold_optimo(Y_test, y_pred)
    conteo_positivos_reales = np.sum(Y_test)
    conteo_positivos_predichos = np.sum((y_pred > optimal_threshold).astype(int))

    final_value = conteo_positivos_reales - conteo_positivos_predichos
    results_list.append(
        SheetResult(
            sheet_name,
            final_value,
            stock_grade,
            optimal_threshold,
            predicted_return,
            None,
            None,
        )
    )
    logging.info(
        f"💰 Valor final de esta hoja: {final_value}, Threshold: {optimal_threshold}, Grado: {stock_grade}"
    )


# Programa principal
def main():
    args = parse_argumentos()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.ERROR)

    valid_sheets = get_valid_sheets(EXCEL_FILE_NAME)
    if not valid_sheets:
        logging.error("No se encontraron hojas válidas en el archivo Excel.")
        return

    logging.info(
        f"📂 Encontramos {len(valid_sheets)} hojas válidas en el archivo Excel\n"
    )

    all_data = load_data(sheets_to_load=valid_sheets, single_sheet=False)

    if all_data is None:
        return

    results = []

    for sheet_name in valid_sheets:
        if sheet_name in all_data:
            process_stock_data(all_data[sheet_name], sheet_name, results)

    # Calcular el rendimiento esperado de cada acción
    predicted_returns = [result.predicted_return for result in results]
    performance_grades = assign_performance_grade(predicted_returns)

    # Asignar una calificación utilizando el valor final de cada acción
    final_value_grades = assign_final_value_grade(
        [result.final_value for result in results]
    )

    for index, result in enumerate(results):
        updated_result = result._replace(
            performance_grade=performance_grades[index],
            final_value_grade=final_value_grades[index],
        )
        results[index] = updated_result

    # Ordenando los resultados
    resultados_ordenados = sorted(results, key=lambda x: x.final_value, reverse=True)

    # Guardar los resultados en un archivo JSON
    store_results_to_json(resultados_ordenados)
    store_results_to_excel(resultados_ordenados)
    print("Resultados guardados en el archivo JSON: stock_results.json")
    print("Resultados guardados en el archivo Excel: stock_results.xlsx")

    mensaje_distribucion_puntaje = mostrar_distribucion_puntaje(results)
    logging.info(mensaje_distribucion_puntaje)

    mensaje_top_stocks = mostrar_top_stocks(resultados_ordenados, valid_sheets)
    print(f"{mensaje_top_stocks}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Sucedió un error: {str(e)}")
