import numpy as np
from excel_analysis.constants import COLUMN_NAMES

def compute_predicted_return(model, X_test):
    """
    Calculate the predicted return based on the model's predictions.

    Parameters:
    - model: Trained MLPRegressor model.
    - X_test: Test data.

    Returns:
    - Predicted return value.
    """
    return np.sum(model.predict(X_test))

def assign_stock_grade(stock_data, y_pred, Y_test):
    """
    Assign a stock grade based on prediction error and volatility.

    Parameters:
    - stock_data: Dataframe containing stock data.
    - y_pred: Model's predictions.
    - Y_test: Actual values.

    Returns:
    - Grade (A, B, C, D, E).
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

def assign_performance_grade(results_list):
    """
    Assign a performance grade based on predicted future performance.

    Parameters:
    - results_list: List of SheetResult namedtuples.

    Returns:
    - Updated list of SheetResults with performance grades.
    """
    predicted_returns = [(result, compute_predicted_return(result.model, result.X_test)) for result in results_list]
    sorted_by_predicted_return = sorted(predicted_returns, key=lambda x: x[1], reverse=True)  # Sort results by predicted returns

    new_results = []
    for rank, (result, _) in enumerate(sorted_by_predicted_return, 1):
        new_results.append(result._replace(performance_grade=rank))
    
    return new_results
