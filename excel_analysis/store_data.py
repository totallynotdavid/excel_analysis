import json
import pandas as pd
from excel_analysis.constants import RESULTS_JSON_FILE_NAME, RESULTS_EXCEL_FILE_NAME


def store_results_to_json(results, filename=RESULTS_JSON_FILE_NAME):
    """
    Guarda los resultados en un archivo JSON.

    Argumentos:
    - results (list): Lista de resultados.
    - filename (str): Nombre del archivo donde guardar los resultados.
    """
    json_data = [result._asdict() for result in results]
    with open(filename, "w") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)


def store_results_to_excel(results, filename=RESULTS_EXCEL_FILE_NAME):
    """
    Guarda los resultados en un archivo Excel.

    Argumentos:
    - results (lista): Lista de resultados.
    - filename (str): Nombre del archivo donde guardar los resultados.
    """
    df = pd.DataFrame([result._asdict() for result in results])
    df.to_excel(filename, index=False)
