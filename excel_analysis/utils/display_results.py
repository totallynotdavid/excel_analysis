def mostrar_distribucion_puntaje(results):
    """
    Mostar la distribución de las calificaciones de rendimiento.
    """
    contador_grados = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for result in results:
        contador_grados[result.performance_grade] += 1

    mensaje_distribucion_grados = "Distribución de las calificaciones de rendimiento:"
    for grade, count in contador_grados.items():
        mensaje_distribucion_grados += f"\n{grade}: {count}"

    total_stocks = len(results)
    for grade, count in contador_grados.items():
        porcentaje = (count / total_stocks) * 100
        mensaje_distribucion_grados += f"\nPorcentaje de stocks con calificación {grade}: {porcentaje:.2f}%"

    return mensaje_distribucion_grados

def mostrar_top_stocks(resultados_ordenados, valid_sheets):
    """
    Mostar los resultados de las acciones en orden de mejor a peor.
    """
    mensaje_stocks = "Resumen de las acciones:"
    mensaje_stocks += "\nLas 10 mejores 📈:"
    for result in resultados_ordenados[:10]:
        mensaje_stocks += f"\n* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f} | Retorno Predicho: {result.predicted_return:.3f} | Rendimiento Grado: {result.performance_grade}"

    if len(valid_sheets) >= 20:
        mensaje_stocks += "\n\nLas 10 peores 📉:"
        for result in resultados_ordenados[-10:]:
            mensaje_stocks += f"\n* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f} | Retorno Predicho: {result.predicted_return:.3f} | Rendimiento Grado: {result.performance_grade}"

    return mensaje_stocks
