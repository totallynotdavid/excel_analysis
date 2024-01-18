import os
import json
import pandas as pd
from excel_analysis.constants import RESULTS_JSON_FILE_NAME, RESULTS_EXCEL_FILE_NAME


def store_results_to_json(results, filename=RESULTS_JSON_FILE_NAME, key="Default"):
    """
    Guarda los resultados en un archivo JSON.

    Argumentos:
    - results (list): Lista de resultados.
    - filename (str): Nombre del archivo donde guardar los resultados.
    - key (str): Key representando el nombre original del archivo.
    """
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file)
    else:
        data = {}

    # Convertir los resultados a un formato de diccionario
    results_dict = [result._asdict() for result in results]

    # Anexar los nuevos resultados bajo la key específica
    data[key] = results_dict

    with open(filename, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Resultados guardados en el archivo JSON {filename} bajo la clave '{key}'.")


def store_results_to_excel(
    results, filename=RESULTS_EXCEL_FILE_NAME, sheet_name="Sheet1"
):
    """
    Guarda los resultados en un archivo Excel.

    Argumentos:
    - results (lista): Lista de resultados.
    - filename (str): Nombre del archivo donde guardar los resultados.
    """
    df = pd.DataFrame([result._asdict() for result in results])

    file_exists = os.path.isfile(filename)

    with pd.ExcelWriter(
        filename, mode="a" if file_exists else "w", engine="openpyxl"
    ) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
