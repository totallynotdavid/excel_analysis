import numpy as np
from excel_analysis.constants import COLUMN_NAMES

def compute_predicted_return(model, X_test):
    """
    Calcula el retorno predicho basado en las predicciones del modelo.

    Parámetros:
    - model: Modelo MLPRegressor entrenado.
    - X_test: Datos de prueba.

    Retorna:
    - Valor de retorno predicho.
    """
    return np.sum(model.predict(X_test))

def assign_stock_grade(stock_data, y_pred, Y_test):
    """
    Asigna una calificación a la acción basada en el error de predicción y la volatilidad.

    Parámetros:
    - stock_data: Dataframe que contiene datos de la acción.
    - y_pred: Predicciones del modelo.
    - Y_test: Valores reales.

    Retorna:
    - Calificación (A, B, C, D, E).
    """
    # Calcular la diferencia entre los valores predichos y reales
    error_prediccion = np.mean(np.abs(y_pred - Y_test))
    
    # Calcular volatilidad (desviación estándar de los rendimientos de la acción)
    retornos_diarios = stock_data[COLUMN_NAMES["price"]].pct_change().dropna()
    retornos_diarios = retornos_diarios.replace([np.inf, -np.inf], np.nan)
    retornos_diarios.interpolate(inplace=True)
    retornos_diarios.fillna(method='ffill', inplace=True)
    retornos_diarios.fillna(method='bfill', inplace=True)

    volatilidad = retornos_diarios.std()

    # Cuantiles de volatilidad
    cuantiles_error = [0.4, 0.5, 0.6, 0.7]
    cuantiles_volatilidad = [0.6, 0.8, 1.0, 1.2]

    if error_prediccion <= cuantiles_error[0] and volatilidad <= cuantiles_volatilidad[0]:
        calificacion = 'A'
    elif error_prediccion <= cuantiles_error[1] and volatilidad <= cuantiles_volatilidad[1]:
        calificacion = 'B'
    elif error_prediccion <= cuantiles_error[2] and volatilidad <= cuantiles_volatilidad[2]:
        calificacion = 'C'
    elif error_prediccion <= cuantiles_error[3] and volatilidad <= cuantiles_volatilidad[3]:
        calificacion = 'D'
    else:
        calificacion = 'E'

    return calificacion

def assign_performance_grade(predicted_returns):
    """
    Asigna una calificación de rendimiento basada en el rendimiento futuro predicho.

    Parámetros:
    - predicted_returns: List of predicted returns for each stock.

    Retorna:
    - grade: Calificación de rendimiento.
    """
    
    # Determine the quantiles of the predicted returns
    q1 = np.quantile(predicted_returns, 0.2)
    q2 = np.quantile(predicted_returns, 0.4)
    q3 = np.quantile(predicted_returns, 0.6)
    q4 = np.quantile(predicted_returns, 0.8)

    grades = []

    # Assign grades based on which quantile a stock's predicted return falls into
    for predicted_return in predicted_returns:
        if predicted_return <= q1:
            grade = 'E'
        elif predicted_return <= q2:
            grade = 'D'
        elif predicted_return <= q3:
            grade = 'C'
        elif predicted_return <= q4:
            grade = 'B'
        else:
            grade = 'A'
        grades.append(grade)
    
    return grades
