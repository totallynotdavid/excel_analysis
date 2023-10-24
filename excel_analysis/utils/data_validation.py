def validar_dataframe(df, columnas_requeridas):
    """
    Valida el DataFrame antes de procesar.
    Retorna True si es válido, False en caso contrario.
    """
    if not set(columnas_requeridas).issubset(df.columns):
        return False

    df.dropna(subset=columnas_requeridas, inplace=True)

    if df.empty or df.isnull().any().any():
        return False

    return True
