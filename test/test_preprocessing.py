import pytest
import pandas as pd
from src.preprocessing import (
    drop_columns,
    fill_nan,
    encode_categorical,
    data_scaling,
    train_test_validation_split,
)


# Test class for `drop_columns` function
class TestDropColumns:
    """
    Test for `drop_columns` function.
    """
    def setup_method(self):
        # Setup the dataframe and columns to drop
        self.df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.columns_to_drop = ["C", "D"]

    def test_drop_columns_valid(self):
        """
        Test `drop_columns` with valid columns to drop.
        """
        columns_to_drop = ["A", "B"]
        for column in columns_to_drop:
            assert column in self.df.columns

    def test_drop_columns_invalid(self):
        """
        Test `drop_columns` with invalid columns to drop.
        """
        columns_to_drop = ["C", "D"]
        with pytest.raises(KeyError):
            drop_columns(self.df, columns_to_drop)

    def test_drop_columns_empty(self):
        """
        Test `drop_columns` with empty columns to drop.
        """
        columns_to_drop = []
        df = drop_columns(self.df, columns_to_drop)
        assert df.shape == (3, 2)


# Test class for `fill_nan` function
class TestFillNaN:
    """
    Test for `fill_nan` function.
    """
    def setup_method(self):
        # Setup the dataframe and fillna option
        self.df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.fillna_option = "Mean"

    def test_fill_nan_valid(self):
        """
        Test `fill_nan` with valid option.
        """
        df = fill_nan(self.df, self.fillna_option)
        assert df.shape == (3, 2)

    def test_fill_nan_invalid(self):
        """
        Test `fill_nan` with invalid option.
        """
        fillna_option = "InvalidOption" or ""
        with pytest.raises(ValueError):
            fill_nan(self.df, fillna_option)


# Test class for `encode_categorical` function
class TestEncodeCategorical:
    """
    Test for `encode_categorical` function.
    """
    def setup_method(self):
        # Setup the dataframe and encoding option
        self.df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.encoding_option = ["Label Encoding", "One-Hot Encoding"]
        self.encoding_columns = ["A", "B"]

    def test_encode_categorical_valid(self):
        """
        Test `encode_categorical` with valid option.
        """
        assert encode_categorical(self.df, self.encoding_option[0], self.encoding_columns).shape == (3, 2)
        assert encode_categorical(self.df, self.encoding_option[1], self.encoding_columns[0]).shape[0] == 3

    def test_encode_categorical_invalid(self):
        """
        Test `encode_categorical` with invalid option.
        """
        encoding_option = "InvalidOption" or ""
        with pytest.raises(ValueError):
            encode_categorical(self.df, encoding_option, self.encoding_columns)


# Test class for `data_scaling` function
class TestDataScaling:
    """
    Test for `data_scaling` function.
    """
    def setup_method(self):
        # Setup the dataframe and scaling option
        self.df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}
        )
        self.scaling_option = ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "Normalizer"]
        self.scaling_columns = ["A", "B"]

    def test_scaling_valid(self):
        """
        Test `data_scaling` with valid option.
        """
        for option in self.scaling_option:
            df = data_scaling(self.df, option, self.scaling_columns)
            assert df.shape == (3, 2)

    def test_scaling_invalid(self):
        """
        Test `data_scaling` with invalid option.
        """
        scaling_option = "InvalidOption" or ""
        with pytest.raises(ValueError):
            data_scaling(self.df, scaling_option, self.scaling_columns)


# Test class for `train_test_validation_split` function
class TestTrainTestValidationSplit:
    """
    Test for `train_test_validation_split` function.
    """
    def setup_method(self):
        # Setup the dataframe and target column
        self.df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.target_column = "A"
        self.test_size = 0.2
        self.random_state = 42

    def test_train_test_validation_split_valid(self):
        """
        Test `train_test_validation_split` with valid option.
        """
        X_train, X_test, y_train, y_test = train_test_validation_split(
            self.df, self.target_column, self.test_size, self.random_state
        )
        assert X_train.shape == (2, 1)
        assert X_test.shape == (1, 1)
        assert y_train.shape == (2,)
        assert y_test.shape == (1,)

    def test_train_test_validation_split_invalid(self):
        """
        Test `train_test_validation_split` with invalid option.
        """
        target_column = "InvalidColumn"
        with pytest.raises(KeyError):
            train_test_validation_split(self.df, target_column, self.test_size, self.random_state)

    def test_random_state_inf(self):
        """
        Test `train_test_validation_split` with inf as seed.
        """
        random_state = float("inf")
        with pytest.raises(ValueError):
            train_test_validation_split(self.df, self.target_column, self.test_size, random_state)

    def test_random_state_nan(self):
        """
        Test `train_test_validation_split` with nan as seed.
        """
        random_state = float("nan")
        with pytest.raises(ValueError):
            train_test_validation_split(self.df, self.target_column, self.test_size, random_state)

    def test_random_state_none(self):
        """
        Test `train_test_validation_split` with None as seed.
        """
        random_state = None
        assert train_test_validation_split(self.df, self.target_column, self.test_size, random_state) is not None
