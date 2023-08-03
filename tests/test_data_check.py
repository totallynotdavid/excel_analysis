import pytest
import pandas as pd
from excel_analysis import data_check as da

def test_load_data():
    df = da.load_data()
    assert isinstance(df, pd.DataFrame)

def test_check_data_size():
    df = da.load_data()
    assert da.check_data_size(df) == 32463

def test_check_data_shape():
    df = da.load_data()
    assert da.check_data_shape(df) == (3607, 9)

def test_check_null_values():
    df = da.load_data()
    expected_null_values = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1], 
                                     index=['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                                             'Momentdiez', 'Momentsetenta', 'Momenttrescerocero', 'Detalle', 'Variación'])
    pd.testing.assert_series_equal(da.check_null_values(df), expected_null_values)

def test_check_top_5_price_counts():
    df = da.load_data()
    expected_top_5 = pd.Series([3, 3, 3, 3, 3], 
                               index=pd.Index([43.125, 25.188, 31.650, 24.335, 24.905], name='Precio'), 
                               name='count')
    pd.testing.assert_series_equal(da.check_top_5_price_counts(df), expected_top_5)

def test_get_column_names():
    df = da.load_data()
    expected_column_names = pd.Index(['Precio', 'Movilveintiuno', 'Movilcincocinco', 'Movilunocuatrocuatro',
                                      'Momentdiez', 'Momentsetenta', 'Momenttrescerocero', 'Detalle', 'Variación'])
    pd.testing.assert_index_equal(da.get_column_names(df), expected_column_names)

def test_normalize_column():
    df = pd.DataFrame({'A': [1, 2, 3]})
    da.normalize_column(df, 'A')
    expected_df = pd.DataFrame({'A': [0.0, 0.5, 1.0]})
    pd.testing.assert_frame_equal(df, expected_df)

def test_get_head():
    df = da.load_data()
    da.normalize_data(df)
    head = da.get_head(df)
    assert isinstance(head, pd.DataFrame)
    assert head.shape == (5, 7)  # Check that we got the first 5 rows of 7 columns