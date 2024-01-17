import numpy as np
from excel_analysis.utils.grading_system import (
    assign_stock_grade,
)
from excel_analysis.models.neural_networks import obtener_threshold_optimo
from excel_analysis.constants import SheetResult


def calcular_calificaciones_y_umbral(df, y_pred, Y_test, price_column, sheet_name):
    # Asignar una calificación a la acción
    stock_grade = assign_stock_grade(df, y_pred, Y_test, price_column)
    predicted_return = np.sum(y_pred)

    # Obtener el umbral (threshold) óptimo
    optimal_threshold = obtener_threshold_optimo(Y_test, y_pred)
    conteo_positivos_reales = np.sum(Y_test)
    conteo_positivos_predichos = np.sum((y_pred > optimal_threshold).astype(int))

    final_value = conteo_positivos_reales - conteo_positivos_predichos

    return SheetResult(
        sheet_name,
        final_value,
        stock_grade,
        optimal_threshold,
        predicted_return,
        None,
        None,
    )
