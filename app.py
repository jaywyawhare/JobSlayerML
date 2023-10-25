from src import *
from src.models import models, eval, training
from src.plot_generator import (
    generate_regression_plots,
    generate_confusion_matrix,
)
from src.preprocessing import (
    drop_columns,
    fill_nan,
    encode_categorical,
    data_scaling,
    train_test_validation_split,
)

import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go


st.set_page_config(layout="wide")


def main():
    global submit_clicked

    st.title("JobSlayerML : Who needs job security?")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
            return None

        st.write(df)
        st.write("Description of the dataset")
        st.write(df.describe())

        st.sidebar.subheader("Specify Parameters")
        target_column = st.sidebar.selectbox("Select the target column", df.columns)
        test_size = st.sidebar.slider("Select the test size", 0.0, 1.0, 0.2, 0.05)
        random_state = st.sidebar.slider("Select the random state", 0, 100, 42, 1)

        st.sidebar.subheader("Specify Model Options")
        model_type = st.sidebar.selectbox(
            "Select the model type", ["Regression", "Classification"]
        )
        comparative_analysis = st.sidebar.checkbox("Comparative Analysis")
        if comparative_analysis:
            model_selector = "Comparative Analysis"
        else:
            model_selector = st.sidebar.selectbox(
                "Select the model", [name for name, model in models(model_type)]
            )


        if model_type == "Regression":
            eval_metrics = st.sidebar.multiselect(
                "Select the evaluation metrics",
                [
                    "MSE",
                    "RMSE",
                    "MAE",
                    "R2",
                    "Adjusted R2",
                    "Explained Variance",
                    "Max Error",
                    "Mean Absolute Percentage Error",
                    "Median Absolute Error",
                    "Mean Squared Log Error",
                    "Mean Poisson Deviance",
                    "Mean Gamma Deviance",
                    "Mean Tweedie Deviance",
                ],
            )
        else:
            eval_metrics = st.sidebar.multiselect(
                "Select the evaluation metrics",
                ["Accuracy", "Precision", "Recall", "F1", "AUC",],
            )
        comparison_metrics = st.sidebar.selectbox(
            "Select the comparison metric", eval_metrics
        )

        st.sidebar.subheader("Specify Preprocessing Options")
        columns_to_drop = st.sidebar.multiselect(
            "Select the columns to drop", df.columns
        )
        fillna_option = st.sidebar.selectbox(
            "Select the fillna option", ["Mean", "Mode", "0"]
        )
        encoding_option = st.sidebar.selectbox(
            "Select the encoding option", ["Label Encoding", "One-Hot Encoding"]
        )
        encoding_columns = st.sidebar.multiselect(
            "Select the columns to encode", df.columns
        )
        scaling_option = st.sidebar.selectbox(
            "Select the scaling option",
            ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "Normalizer"],
        )
        scaling_columns = st.sidebar.multiselect(
            "Select the columns to scale", df.columns
        )


        if not comparative_analysis:
            st.sidebar.subheader("Specify Plot Options")
            if model_type == "Regression":
                plot_options = [
                    "Regression Plot",
                    "Scatter Matrix",
                    "Box Plot",
                    "Histogram",
                ]
            else:
                plot_options = [
                    "Confusion Matrix",
                    "Heatmap",
                    "Scatter Matrix",
                    "Box Plot",
                    "Histogram",
                    "Bar Plot",
                    "Count Plot",
                    "Violin Plot",
                ]
            plot_selector = st.sidebar.multiselect(
                "Select the plots to generate", plot_options
            )

        submit_clicked = st.sidebar.button("Submit")

        if not submit_clicked:
            return None

        df = drop_columns(df, columns_to_drop)
        df = pd.DataFrame(df)
        df = fill_nan(df, fillna_option)
        df = encode_categorical(df, encoding_option, encoding_columns)
        df, scaler = data_scaling(df, scaling_option, scaling_columns)
        X_train, X_test, y_train, y_test = train_test_validation_split(
            df, target_column, test_size, random_state
        )


        if model_selector == "Comparative Analysis":
            training(
                model_type,
                model_selector,
                X_train,
                y_train,
                X_test,
                y_test,
                eval_metrics,
                comparison_metrics,
            )
        else:
            y_test, y_pred, best_model = training(
                model_type,
                model_selector,
                X_train,
                y_train,
                X_test,
                y_test,
                eval_metrics,
                comparison_metrics,
            )

            if "Regression Plot" in plot_selector:
                generate_regression_plots(X_train, X_test, y_train, y_test, best_model)
            if "Confusion Matrix" in plot_selector:
                generate_confusion_matrix(y_test, y_pred)
            


if __name__ == "__main__":
    main()
