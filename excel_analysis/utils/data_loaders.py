import logging
import pandas as pd

from excel_analysis.constants import EXCEL_FILE_NAME, INDEX_COLUMN

def get_valid_sheets(file_name):
    """
    Retrieve valid sheets from the provided Excel file. A sheet is considered valid if it contains the INDEX_COLUMN.

    Parameters:
    - file_name (str): Path to the Excel file.

    Returns:
    - list: List of valid sheet names.
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
    Load data from the provided Excel file.

    Parameters:
    - file_name (str): Path to the Excel file.
    - sheets_to_load (list): List of specific sheet names to load.
    - single_sheet (bool): Whether to load only a single sheet.

    Returns:
    - dict: Dictionary containing data from the Excel sheets.
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
