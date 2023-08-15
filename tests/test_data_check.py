import pytest
import pandas as pd
import numpy as np
from excel_analysis import stock_processing as da

def test_load_data():
    df = da.load_data(single_sheet=True)
    assert isinstance(df, pd.DataFrame)

def test_check_data_size():
    df = da.load_data(single_sheet=True)
    assert da.check_data_size(df) == 32463

def test_check_data_shape():
    df = da.load_data(single_sheet=True)
    assert da.check_data_shape(df) == (3607, 9)

def test_check_null_values():
    df = da.load_data(single_sheet=True)
    expected_null_values = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1], 
                                     index=['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                                             'Momentdiez', 'Momentsetenta', 'Momenttrescerocero', 'Detalle', 'Variación'])
    pd.testing.assert_series_equal(da.check_null_values(df), expected_null_values)

def test_check_top_5_price_counts():
    df = da.load_data(single_sheet=True)
    expected_top_5 = pd.Series([3, 3, 3, 3, 3], 
                               index=pd.Index([43.125, 25.188, 31.650, 24.335, 24.905], name='Precio'), 
                               name='count')
    price_column = da.COLUMN_NAMES["price"]
    pd.testing.assert_series_equal(da.check_top_5_price_counts(df, price_column), expected_top_5)

def test_get_column_names():
    df = da.load_data(single_sheet=True)
    expected_column_names = pd.Index(['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                                      'Momentdiez', 'Momentsetenta', 'Momenttrescerocero', 'Detalle', 'Variación'])
    pd.testing.assert_index_equal(da.get_column_names(df), expected_column_names)

def test_normalize_column():
    df = pd.DataFrame({'A': [1, 2, 3]})
    da.normalize_column(df, 'A')
    expected_df = pd.DataFrame({'A': [0.0, 0.5, 1.0]})
    pd.testing.assert_frame_equal(df, expected_df)

def test_get_head():
    df = da.load_data(single_sheet=True)
    da.normalize_data(df)
    price_column = da.COLUMN_NAMES["price"]
    features = da.COLUMN_NAMES["features"]
    head = da.get_head(df, price_column, features)
    assert isinstance(head, pd.DataFrame)
    assert head.shape == (5, 7)  # Comprobar que tenemos 5 filas de las 7 columnas que queremos

def test_check_dtypes():
    df = pd.DataFrame({
        'Precio': [1.0, 2.0],
        'Movilveintiuno': [3.0, 4.0],
        'Movilcincocinco': [5.0, 6.0],
        'Movilunocuatrocuatro': [7.0, 8.0],
        'Momentdiez': [9.0, 10.0],
        'Momentsetenta': [11.0, 12.0],
        'Momenttrescerocero': [13.0, 14.0],
        'Detalle': [15.0, 16.0],
        'Variación': [17.0, 18.0]
    })
    # Should work as all columns are float64
    da.ensure_float64(df)

    # ValueError as one column is not float64
    df['Detalle'] = df['Detalle'].astype('int64')
    with pytest.raises(ValueError, match="Todas las columnas deben ser de tipo float64"):
        da.ensure_float64(df)

def test_normalize_data():
    df = pd.DataFrame({
        'Precio': [1, 2, 3],
        'Movilveintiuno': [4, 5, 6],
        'Movilcincocinco': [7, 8, 9],
        'Movilunocuatrocuatro': [10, 11, 12],
        'Momentdiez': [13, 14, 15],
        'Momentsetenta': [16, 17, 18],
        'Momenttrescerocero': [19, 20, 21]
    })
    da.normalize_data(df)
    for column in ['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro', 'Momentdiez', 'Momentsetenta', 'Momenttrescerocero']:
        assert df[column].min() == 0.0
        assert df[column].max() == 1.0

# Asegurarnos de que la división entre training y test data es correcta
def test_get_training_and_test_data():
    df = da.load_data(single_sheet=True)
    da.normalize_data(df)
    df = df[pd.to_numeric(df['Detalle'], errors='coerce').notnull()]
    df['Detalle'] = df['Detalle'].astype('float')
    X, Y, X2, Y2 = da.get_training_and_test_data(df)

    # Revisando para los valores de X
    expected_X = np.array([
        [0.00287822, 0.00710203, 0.01279801, 0.63256638, 0.14308482, 0.10067925],
        [0.00286914, 0.00679073, 0.01270777, 0.39075603, 0.14193188, 0.10809786],
        [0.00271699, 0.0064915, 0.01260582, 0.2710176, 0.1132703, 0.1084489],
        [0.00257997, 0.0061703, 0.01248572, 0.05708146, 0.08028511, 0.10163773],
        [0.00256511, 0.00586935, 0.0123876, 0.24765987, 0.11286448, 0.10970468]
    ])

    expected_X2 = np.array([
        [0.37546853, 0.3895117, 0.40677795, 0.56945149, 0.38200623, 0.4861611],
        [0.37878119, 0.38848497, 0.40744651, 0.58059124, 0.36655321, 0.47122028],
        [0.38209853, 0.38763101, 0.40815859, 0.69000358, 0.37896467, 0.47974739],
        [0.38596064, 0.38689338, 0.40894673, 0.65699767, 0.41279831, 0.48731584],
        [0.38858546, 0.38639876, 0.40973961, 0.69161736, 0.39928953, 0.50044671]
    ])

    np.testing.assert_array_almost_equal(X[0:5], expected_X)
    np.testing.assert_array_almost_equal(X2[0:5], expected_X2)

    # Revisando para los valores de Y
    expected_Y = np.array([1., 1., 1., 0., 1.])
    expected_Y2 = np.array([1., 0., 0., 0., 0.])

    np.testing.assert_array_almost_equal(Y[0:5], expected_Y)
    np.testing.assert_array_almost_equal(Y2[0:5], expected_Y2)