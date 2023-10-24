def display_grade_distribution(results):
    """
    Mostar la distribución de las calificaciones de rendimiento.
    """
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    
    for result in results:
        grade_counts[result.performance_grade] += 1

    print("\nDistribución de las calificaciones de rendimiento:")
    for grade, count in grade_counts.items():
        print(f"{grade}: {count}")

    total_stocks = len(results)
    for grade, count in grade_counts.items():
        percentage = (count / total_stocks) * 100
        print(f"Porcentaje de stocks con calificación {grade}: {percentage:.2f}%")

def display_top_stocks(sorted_results, valid_sheets):
    """
    Mostar los resultados de las acciones en orden de mejor a peor.
    """
    print("\nResumen de las acciones:")

    print("\nLas 10 mejores 📈:")
    for result in sorted_results[:10]:
        print(f"* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f} | Retorno Predicho: {result.predicted_return:.3f} | Rendimiento Grado: {result.performance_grade}")

    if len(valid_sheets) >= 20:
        print("\nLas 10 peores 📉:")
        for result in sorted_results[-10:]:
            print(f"* Hoja: {result.sheet_name} | Valor: {result.final_value} | Grado: {result.grade} | Threshold: {result.optimal_threshold:.3f} | Retorno Predicho: {result.predicted_return:.3f} | Rendimiento Grado: {result.performance_grade}")
