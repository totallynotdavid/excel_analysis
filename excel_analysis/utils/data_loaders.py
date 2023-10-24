import logging
import pandas as pd

from excel_analysis.constants import EXCEL_FILE_NAME, INDEX_COLUMN

def get_valid_sheets(file_name):
    """
    Recupera las hojas válidas del archivo Excel proporcionado. Una hoja se considera válida si contiene el INDEX_COLUMN.

    Parámetros:
    - file_name (str): Ruta al archivo Excel.

    Retorna:
    - list: Lista de nombres de hojas válidas. Solo las hojas de esta lista serán analizadas.
    """
    try:
        headers = pd.read_excel(file_name, sheet_name=None, engine='openpyxl', nrows=0)
        valid_sheets = [sheet for sheet, df in headers.items() if INDEX_COLUMN in df.columns]
        
        if not valid_sheets:
            logging.error(f"El archivo '{file_name}' no tiene hojas válidas con la columna '{INDEX_COLUMN}'.")
            return []

        return valid_sheets

    except Exception as e:
        logging.error(f"No se puede leer las cabeceras del archivo '{file_name}'. Error: {str(e)}")
        return []

def load_data(file_name=EXCEL_FILE_NAME, sheets_to_load=None, single_sheet=False):
    """
    Carga datos del archivo Excel proporcionado.

    Parámetros:
    - file_name (str): Ruta al archivo Excel.
    - sheets_to_load (list): Lista de nombres de hojas específicas para cargar.
    - single_sheet (bool): Si se debe cargar solo una hoja.

    Retorna:
    - dict: Diccionario que contiene datos de las hojas de Excel.
    """
    try:
        valid_sheets = get_valid_sheets(file_name)

        if not valid_sheets:
            return None

        if sheets_to_load:
            valid_sheets = [sheet for sheet in valid_sheets if sheet in sheets_to_load]

        data = pd.read_excel(file_name, sheet_name=valid_sheets, engine='openpyxl', index_col=INDEX_COLUMN)

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
