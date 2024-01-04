def mostrar_distribucion_puntaje(results):
    """
    Mostar la distribución de las calificaciones de rendimiento.
    """
    contador_grados = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for result in results:
        contador_grados[result.performance_grade] += 1

    mensaje_distribucion_grados = "Distribución de las calificaciones de rendimiento:"
    for grade, count in contador_grados.items():
        mensaje_distribucion_grados += f"\n{grade}: {count}"

    total_stocks = len(results)
    for grade, count in contador_grados.items():
        porcentaje = (count / total_stocks) * 100
        mensaje_distribucion_grados += (
            f"\nPorcentaje de stocks con calificación {grade}: {porcentaje:.2f}%"
        )

    return mensaje_distribucion_grados


def crear_mensaje_stock(result, prefix="*"):
    """
    Crear un mensaje para una acción dada.

    Parámetros:
    - result: Objeto de resultado de la acción.
    - prefix: Prefijo para el mensaje.

    Retorna:
    - Mensaje de la acción.
    """
    return (
        f"{prefix} Hoja: {result.sheet_name} | Valor: {result.final_value} | "
        f"Threshold: {result.optimal_threshold:.3f} | "
        f"Retorno Predicho: {result.predicted_return:.3f} | "
        f"Grado (Vol. & Error): {result.grade} | "
        f"Grado (Rendimiento): {result.performance_grade} | "
        f"Grado (Valor Final): {result.final_value_grade}"
    )


def mostrar_top_stocks(resultados_ordenados, valid_sheets):
    """
    Mostar los resultados de las acciones en orden de mejor a peor.
    """
    top_10 = "\n".join(
        crear_mensaje_stock(result) for result in resultados_ordenados[:10]
    )
    peores_10 = "\n".join(
        crear_mensaje_stock(result) for result in resultados_ordenados[-10:]
    )

    mensaje_stocks = "Resumen de las acciones:"
    mensaje_stocks += "\nLas 10 mejores 📈:\n" + top_10

    if len(valid_sheets) >= 20:
        mensaje_stocks += "\n\nLas 10 peores 📉:\n" + peores_10

    return mensaje_stocks
