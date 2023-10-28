from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st


def drop_columns(df, columns_to_drop):
    """
    Function to drop columns from the dataframe.

    Args:
        df: The dataframe to drop columns from.
        columns_to_drop: The columns to drop.

    Returns:
        The dataframe with the columns dropped.
    """
    df = pd.DataFrame(df)
    for column in columns_to_drop:
        df = df.drop(columns=[column], axis=1)
    return df


def fill_nan(df, fillna_option):
    """
    Function to fill NaN values in the dataframe.

    Args:
        df: The dataframe to fill NaN values in.
        fillna_option: The option to fill NaN values with.

    Returns:
        The dataframe with the NaN values filled.

    """
    df = df.fillna(df.mean())
    if fillna_option == "Mean":
        df = df.fillna(df.mean())
    elif fillna_option == "Mode":
        df = df.fillna(df.mode())
    elif fillna_option == "0":
        df = df.fillna(0)
    else:
        raise ValueError("Invalid option for filling NaN values.")
    return df


def encode_categorical(df, encoding_option, encoding_columns):
    """
    Function to encode categorical columns in the dataframe.

    Args:
        df: The dataframe to encode categorical columns in.
        encoding_option: The option to encode categorical columns with.
        encoding_columns: The columns to encode.

    Returns:
        The dataframe with the categorical columns encoded.
    """
    df = pd.DataFrame(df)
    if encoding_option == "Label Encoding":
        le = LabelEncoder()
        df[encoding_columns] = df[encoding_columns].apply(le.fit_transform)
    elif encoding_option == "One-Hot Encoding":
        df[encoding_columns] = df[encoding_columns].astype(str)
        encoder = OneHotEncoder()
        one_hot = encoder.fit_transform(df[[encoding_columns]])
        one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot, columns=encoder.get_feature_names_out([encoding_columns]))
        df = pd.concat([df, one_hot_df], axis=1)
        df = df.drop(columns=[encoding_columns])
    else:
        raise ValueError("Invalid option for encoding categorical columns.")
    return df


def data_scaling(df, scaling_option, scaling_columns):
    """
    Function to scale the data in the dataframe.

    Args:
        df: The dataframe to scale.
        scaling_option: The option to scale the data with.
        scaling_columns: The columns to scale.

    Returns:
        The dataframe with the data scaled.
    """
    df = pd.DataFrame(df)
    if scaling_option == "Standard Scaler":
        scaler = StandardScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    elif scaling_option == "MinMax Scaler":
        scaler = MinMaxScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    elif scaling_option == "Robust Scaler":
        scaler = RobustScaler()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    elif scaling_option == "Normalizer":
        scaler = Normalizer()
        df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    else:
        raise ValueError("Invalid option for scaling data.")
    return df


def train_test_validation_split(df, target_column, test_size, random_state):
    """
    Function to split the data into training, testing and validation sets.

    Args:
        df: The dataframe to split.
        target_column: The target column.
        test_size: The size of the testing set.
        random_state: The random state.

    Returns:
        The training, testing and validation sets.
    """
    df = pd.DataFrame(df)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
