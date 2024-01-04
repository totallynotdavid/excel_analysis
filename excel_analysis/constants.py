from collections import namedtuple

EXCEL_FILE_NAME = "Prueba3.xlsx"
RESULTS_JSON_FILE_NAME = "stock_results.json"
RESULTS_EXCEL_FILE_NAME = "stock_results.xlsx"
INDEX_COLUMN = "FECHA"
TRAIN_TEST_SPLIT_RATIO = 0.8

COLUMN_NAMES = {
    "price": "PX_LAST",
    "features": [
        "Inflacion_Mensual",
        "PMI_MANUFACTURERO",
        "PMI_COMPOSITE",
        "ISM_MANUFACTURERO",
        "INFLACION",
        "TASA_INTERES",
        "IPC_SINALIMENTOS_ENERGIA",
        "PRODUCCIÓN_INDUSTRIAL",
    ],
    "detail": "DETALLE",
}

SheetResult = namedtuple(
    "SheetResult",
    [
        "sheet_name",
        "final_value",
        "grade",
        "optimal_threshold",
        "predicted_return",
        "performance_grade",
        "final_value_grade",
    ],
    defaults=[None],
)
