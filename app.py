import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
from collections import defaultdict

def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
            return None
    else:
        return None

def perform_regression(X_train, y_train, X_test, y_test, selected_model):
    model = selected_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

def perform_classification(X_train, y_train, X_test, y_test, selected_model):
    model = selected_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def main():
    st.title("AutoML: Automated Machine Learning")

    df = load_data()

    if df is None:
        return

    st.sidebar.header("Data Preprocessing Options")
    fillna_option = st.sidebar.radio("Fill NaN with:", ("Mean", "Mode", "0"))

    task = st.sidebar.radio("Choose a Task:", ("Regression", "Classification"))

    if fillna_option == "Mean":
        fill_value = df.mean()
    elif fillna_option == "Mode":
        fill_value = df.mode().iloc[0]
    else:
        fill_value = 0

    df.fillna(fill_value, inplace=True)

    target_column = st.sidebar.selectbox("Select the Target Column:", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.sidebar.header("Choose a Model")
    model_selector = st.sidebar.selectbox("Select a model:", sorted([name for name, _ in all_estimators(type_filter=['regressor' if task == 'Regression' else 'classifier'])]))
    selected_model = [est for name, est in all_estimators(type_filter=['regressor' if task == 'Regression' else 'classifier']) if name == model_selector][0]

    if task == "Regression":
        st.header("Regression Task")
        st.write(f"Selected Model: {model_selector}")

        model, mse = perform_regression(X_train, y_train, X_test, y_test, selected_model)
        st.write(f"Mean Squared Error: {mse}")

    elif task == "Classification":
        st.header("Classification Task")
        st.write(f"Selected Model: {model_selector}")

        model, accuracy = perform_classification(X_train, y_train, X_test, y_test, selected_model)
        st.write(f"Accuracy: {accuracy}")

    st.subheader("Best Model")
    st.write("The best model for the dataset is:", model_selector)

if __name__ == "__main__":
    main()
