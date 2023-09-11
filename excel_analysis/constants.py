from collections import namedtuple

EXCEL_FILE_NAME = 'Hoja_1.xlsx'
COLUMN_NAMES = {
    "price": "PX_LAST",
    "features": ['PX_MID', 'PX_LOW', 'PX_HIGH', 'PX_OPEN', 'RSI_14D', 'RSI_30D', 'RSI_9D', 'RSI_3D'],
    "detail": 'DETALLE',
}
INDEX_COLUMN = 'FECHA'
TRAIN_TEST_SPLIT_RATIO = 0.8
SheetResult = namedtuple("SheetResult", ["sheet_name", "final_value"])