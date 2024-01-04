import pandas as pd
import numpy as np
from excel_analysis.constants import COLUMN_NAMES


def ensure_float64(df, columns_to_check):
    """
    Asegura que las columnas especificadas en el DataFrame sean de tipo float64.
    """
    if not all(df[columns_to_check].dtypes == "float64"):
        raise ValueError(f"Las columnas {columns_to_check} deben ser de tipo float64")


def handle_non_numeric_values(df, columns_to_check):
    """
    Convertir valores no numéricos en el DataFrame a NaN y asegurarse de que las columnas sean de tipo float64.
    """
    for column in columns_to_check:
        is_non_numeric = pd.to_numeric(df[column], errors="coerce").isna()

        if is_non_numeric.sum() > 0:
            # Handle non-numeric values found in the column
            non_numeric_rows = df[is_non_numeric][column]
            for idx, value in non_numeric_rows.items():
                print(f"Fila: {idx}, Valor: {value}")

        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].dtype != "float64":
            df[column] = df[column].astype("float64")

    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)


def normalize_data(df):
    """
    Normalizar los datos para las columnas especificadas en el DataFrame.
    """
    columns_to_normalize = [COLUMN_NAMES["price"]] + COLUMN_NAMES["features"]
    for column in columns_to_normalize:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )


def dividir_datos_entrenamiento_prueba(df, train_test_split_ratio):
    """
    Dividir los datos en conjuntos de entrenamiento y prueba basados en el ratio definido en constants.
    """
    train_size = int(train_test_split_ratio * len(df))
    datos_entrenamiento = df.iloc[:train_size]
    datos_prueba = df.iloc[train_size:]

    feature_cols = COLUMN_NAMES["features"]
    X_train = datos_entrenamiento[feature_cols].values
    Y_train = datos_entrenamiento[COLUMN_NAMES["detail"]].values.astype("float")

    X_test = datos_prueba[feature_cols].values
    Y_test = datos_prueba[COLUMN_NAMES["detail"]].values.astype("float")

    return X_train, Y_train, X_test, Y_test
