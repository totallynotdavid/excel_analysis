def mostrar_distribucion_puntaje(results):
    """
    Mostar la distribución de las calificaciones de rendimiento.
    """
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for result in results:
        grade_counts[result.performance_grade] += 1

    grade_distribution_msg = "Distribución de las calificaciones de rendimiento:"
    for grade, count in grade_counts.items():
        grade_distribution_msg += f"\n{grade}: {count}"

    total_stocks = len(results)
    for grade, count in grade_counts.items():
        percentage = (count / total_stocks) * 100
        grade_distribution_msg += f"\n\nPorcentaje de stocks con calificación {grade}: {percentage:.2f}%"

    return grade_distribution_msg

def mostrar_top_stocks(resultados_ordenados, valid_sheets):
    """
    Mostar los resultados de las acciones en orden de mejor a peor.
    """
    stocks_msg = "Resumen de las acciones:"
    stocks_msg += "\nLas 10 mejores 📈:"
    for result in resultados_ordenados[:10]:
        stocks_msg += f"\n* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f} | Retorno Predicho: {result.predicted_return:.3f} | Rendimiento Grado: {result.performance_grade}"

    if len(valid_sheets) >= 20:
        stocks_msg += "\nLas 10 peores 📉:"
        for result in resultados_ordenados[-10:]:
            stocks_msg += f"\n* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f} | Retorno Predicho: {result.predicted_return:.3f} | Rendimiento Grado: {result.performance_grade}"

    return stocks_msg
