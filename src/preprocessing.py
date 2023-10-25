from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st


def drop_columns(df, columns_to_drop):
    df = pd.DataFrame(df)
    for column in columns_to_drop:
        df = df.drop(columns=[column], axis=1)
    return df


def fill_nan(df, fillna_option):
    df = df.fillna(df.mean())
    if fillna_option == "Mean":
        df = df.fillna(df.mean())
    elif fillna_option == "Mode":
        df = df.fillna(df.mode())
    elif fillna_option == "0":
        df = df.fillna(0)
    return df


def encode_categorical(df, encoding_option, encoding_columns):
    df = pd.DataFrame(df)
    if encoding_option == "Label Encoding":
        le = LabelEncoder()
        df[encoding_columns] = df[encoding_columns].apply(le.fit_transform)
    elif encoding_option == "One-Hot Encoding":
        oneHot = pd.get_dummies(df[encoding_columns])
        df = pd.concat([df, oneHot], axis=1)
        df = df.drop(columns=encoding_columns, axis=1)
    return df


def data_scaling(df, scaling_option, scaling_columns):
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
    return df, scaler


def train_test_validation_split(df, target_column, test_size, random_state):
    df = pd.DataFrame(df)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
