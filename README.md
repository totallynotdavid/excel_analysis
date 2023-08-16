# Análisis de archivos XSLX con Python

Este proyecto analiza datos de acciones almacenados en archivos Excel, utilizando técnicas avanzadas de regresión para obtener perspectivas predictivas.

## Antes de ejecutar

### Prerrequisitos

1. Asegúrate de tener Python instalado.
2. El proyecto utiliza Poetry para la gestión de dependencias. Si aún no has instalado Poetry:
   - Para usuarios de Windows:

    ```
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

### Configuración del proyecto

1. Activa el entorno de Poetry:

   ```
   poetry shell
   ```

2. Instala las dependencias del proyecto:

   ```
   poetry install
   ```

3. Ejecuta el proyecto:

   ```
   poetry run start
   ```

### Ejecutando sin Poetry

Si prefieres no usar Poetry, también puedes ejecutar el módulo directamente con Python:

```
python -m excel_analysis.stock_processing
```

## Estructura de la hoja Excel

El programa espera que las hojas Excel tengan la siguiente estructura:

| FECHA     | Precio | Movilveintiuno | Movilcincocinco | Movilunocuatrocuatro | Momentdiez | Momentsetenta | Momenttrescerocero | Detalle | Variación |
|-----------|--------|-----------------|-----------------|-----------------------|------------|---------------|--------------------|---------|-----------|
| 7/11/2008 | 3.509  | 3.61            | 4.42            | 5.54                  | 106.69     | 62.72         | 68.15              | 1       | 2.48%     |
| 10/11/2008| 3.424  | 3.61            | 4.36            | 5.53                  | 95.96      | 62.57         | 70.11              | 1       | 1.15%     |
| 11/11/2008| 3.385  | 3.58            | 4.31            | 5.51                  | 90.65      | 59.00         | 70.20              | 1       | 5.16%     |
| 12/11/2008| 3.219  | 3.56            | 4.26            | 5.49                  | 81.16      | 54.89         | 68.40              | 0       | -6.53%    |
