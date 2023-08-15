from collections import namedtuple

EXCEL_FILE_NAME = 'Base2.xlsx'
COLUMN_NAMES = {
    "price": "Precio",
    "features": ['Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro', 'Momentdiez', 'Momentsetenta', 'Momenttrescerocero'],
    "detail": 'Detalle',
}
SheetResult = namedtuple("SheetResult", ["sheet_name", "final_value"])