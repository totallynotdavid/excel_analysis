import pandas as pd

from excel_analysis.constants import COLUMN_NAMES, TRAIN_TEST_SPLIT_RATIO
from excel_analysis.utils.data_preprocessors import (
    handle_non_numeric_values,
    normalize_data,
    ensure_float64,
    dividir_datos_entrenamiento_prueba,
)
from excel_analysis.models.neural_networks import entrenar_regresor_mlp


def entrenar_y_predecir(df, columnas_requeridas):
    """
    Entrena el modelo y predice usando el dataframe.
    Retorna el modelo, las predicciones, y Y_test.
    """
    handle_non_numeric_values(df, columnas_requeridas)
    normalize_data(df)
    df = df[pd.to_numeric(df[COLUMN_NAMES["detail"]], errors="coerce").notnull()]
    df.loc[:, COLUMN_NAMES["detail"]] = df[COLUMN_NAMES["detail"]].astype("float")
    ensure_float64(df, columnas_requeridas)

    X_train, Y_train, X_test, Y_test = dividir_datos_entrenamiento_prueba(
        df, TRAIN_TEST_SPLIT_RATIO
    )

    if len(X_train) == 0 or len(Y_train) == 0:
        return None, None, None

    modelo_red_neuronal = entrenar_regresor_mlp(X_train, Y_train)
    y_pred = modelo_red_neuronal.predict(X_test)
    return modelo_red_neuronal, y_pred, Y_test
