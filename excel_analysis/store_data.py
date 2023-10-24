import json
from excel_analysis.constants import JSON_FILE_NAME

def store_results_to_json(results, filename=JSON_FILE_NAME):
    """
    Guarda los resultados en un archivo JSON.

    Args:
    - results (list): Lista de resultados.
    - filename (str): Nombre del archivo donde guardar los resultados.

    Returns:
    - None
    """
    json_data = [result._asdict() for result in results]
    with open(filename, 'w') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)