from collections import namedtuple

EXCEL_FILE_NAME = 'Prueba3.xlsx'
COLUMN_NAMES = {
    "price": "PX_LAST",
    "features": ['Inflacion_Mensual', 'PMI_MANUFACTURERO', 'PMI_COMPOSITE', 'ISM_MANUFACTURERO', 'INFLACION', 'TASA_INTERES', 'IPC_SINALIMENTOS_ENERGIA', 'PRODUCCIÓN_INDUSTRIAL'],
    "detail": 'DETALLE',
}
INDEX_COLUMN = 'FECHA'
TRAIN_TEST_SPLIT_RATIO = 0.8
SheetResult = namedtuple("SheetResult", ["sheet_name", "final_value"])