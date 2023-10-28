from src import *
from src.models import (
    models, 
    eval, 
    training, 
    )
from src.plot_generator import (
    generate_regression_plot,
    generate_confusion_matrix,
    generate_heatmap,
    generate_distribution_plot,
    generate_pca_plot,
    generate_2d_tsne,
    generate_3d_tsne,
    generate_2d_umap,
    generate_3d_umap,
    generate_roc_curve,
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
import numpy as np


st.set_page_config(layout="wide")

def main():
    global submit_clicked

    st.title("JobSlayerML: Because Engineers Shouldn't Settle for Job Security!")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
            return None

        st.write(df)

        st.sidebar.subheader("Select the options below")

        target_column = st.sidebar.selectbox("Select the target column", df.columns)

        drop_columns_checkbox = st.sidebar.checkbox("Drop Columns")
        if drop_columns_checkbox:
            columns_to_drop = st.sidebar.multiselect(
                "Select the columns to drop", df.columns
            )

        fillna_checkbox = st.sidebar.checkbox("Fill NaN")
        if df.isnull().values.any() and fillna_checkbox is False:
            st.sidebar.error("The dataset contains NaN values, please fill them")
        
        if fillna_checkbox:
            fillna_option = st.sidebar.selectbox(
                "Select the fill NaN option",
                ["Mean", "Median", "Mode", "0"],
            )

        exploratory_data_analysis_checkbox = st.sidebar.checkbox(
            "Exploratory Data Analysis"
        )
        slider = None
        if exploratory_data_analysis_checkbox:
            slider = st.sidebar.slider(
                "Select the number of components to keep", 1, len(df.columns), 1, 1
            )

        reduced_dimensionality_visualization_checkbox = st.sidebar.checkbox(
            "Reduced Dimensionality Visualization"
        )

        encode_checkbox = st.sidebar.checkbox("Categorical Encoding")
        if encode_checkbox:
            encoding_option = st.sidebar.selectbox(
                "Select the encoding option", ["Label Encoding", "One-Hot Encoding"]
            )
            if encoding_option == "Label Encoding":
                encoding_columns = st.sidebar.multiselect(
                    "Select the columns to encode", df.columns
                )
            elif encoding_option == "One-Hot Encoding":
                encoding_columns = st.sidebar.multiselect(
                    "Select the columns to encode", df.columns
                )
                if target_column in encoding_columns:
                    st.sidebar.error(
                        "The target column cannot be selected for One-Hot Encoding"
                    )
                    return None

        scale_checkbox = st.sidebar.checkbox("Scaling Data")
        if scale_checkbox:
            scaling_option = st.sidebar.selectbox(
                "Select the scaling option",
                ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "Normalizer"],
            )
            scaling_columns = st.sidebar.multiselect(
                "Select the columns to scale", df.columns
            )

        train_test_split_checkbox = st.sidebar.checkbox("Train Test Split")
        if train_test_split_checkbox is False:
            st.sidebar.warning(
                "Please note that the model will be trained on the entire dataset"
            )
            test_size = 0.01
            random_state = 42
            
        if train_test_split_checkbox:
            test_size = st.sidebar.slider("Select the percentage of test data", 0.01, 0.99, 0.20, 0.01)
            random_state = st.sidebar.slider("Select the random state", 0, 100, 42, 1)

        model_type = st.sidebar.radio(
            "Select the model type", ("Regression", "Classification")
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
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "AUC",
                ],
            )
        comparison_metrics = st.sidebar.selectbox(
            "Select the comparison metric", eval_metrics
        )

        model_selection_radio = st.sidebar.radio(
            "Model Selection", ("Comparative Analysis", "Individual Model Selection")
        )

        if model_selection_radio == "Individual Model Selection":
            model_evaluation_checkbox = st.sidebar.checkbox("Model Evaluation")
            if model_evaluation_checkbox:
                model_selector = st.sidebar.selectbox(
                    "Select the model", [name for name, model in models(model_type)]
                )
                model = [
                    est for name, est in models(model_type) if name == model_selector
                ][0]

                hyperparameter_tuning_checkbox = st.sidebar.checkbox(
                    "Hyperparameter Tuning"
                )
                if hyperparameter_tuning_checkbox:
                    st.sidebar.warning(
                        "Hyperparameter tuning is a time consuming process, please be patient and USE YOUR OWN PC!"
                    )

            plot_generation_checkbox = st.sidebar.checkbox("Plot Generation")

            if plot_generation_checkbox and model_type == "Regression":
                plot_selector = "Regression Plot"

            elif plot_generation_checkbox and model_type == "Classification":
                plot_selector = st.sidebar.multiselect(
                    "Select the plots to generate", ["Confusion Matrix", "ROC Curve"]
                )

        else:
            model_selection_radio = "Comparative Analysis"

        if exploratory_data_analysis_checkbox:
            st.markdown(
                "<h2 style='text-align: center;'>Exploratory Data Analysis Results</a></h2>",
                unsafe_allow_html=True,
            )

            st.subheader("Correlation Heatmap")
            generate_heatmap(df)

            st.subheader("Distribution Plot")
            generate_distribution_plot(df)

            st.subheader("Component Analysis")
            generate_pca_plot(df, target_column)

        if reduced_dimensionality_visualization_checkbox:
            st.markdown(
                "<h2 style='text-align: center;'>Reduced Dimensionality Visualization Results</a></h2>",
                unsafe_allow_html=True,
            )
            st.subheader("2D t-SNE")
            generate_2d_tsne(df, target_column)

            st.subheader("2D UMAP")
            generate_2d_umap(df, target_column)

            st.subheader("3D t-SNE")
            generate_3d_tsne(df, target_column)

            st.subheader("3D UMAP")
            generate_3d_umap(df, target_column)

        submit_clicked = st.sidebar.button("Submit")

        if model_type == "Regression" and not np.issubdtype(
            df[target_column].dtype, np.number
        ):
            st.sidebar.warning(
                "The target column is not numerical. Please select a different column."
            )

        if model_type == "Classification" and np.issubdtype(
            df[target_column].dtype, np.number
        ):
            st.sidebar.warning(
                "The target column is numerical. Please categorically encode the target column."
            )

        if not submit_clicked:
            return None

        if drop_columns_checkbox:
            df = drop_columns(df, columns_to_drop)
        if fillna_checkbox:
            df = fill_nan(df, fillna_option)
        if encode_checkbox:
            df = encode_categorical(df, encoding_option, encoding_columns)
        if scale_checkbox:
            df = data_scaling(df, scaling_option, scaling_columns)
        X_train, X_test, y_train, y_test = train_test_validation_split(
            df, target_column, test_size, random_state
        )

        if slider is None:
            slider = X_train.shape[1]

        if model_selection_radio == "Comparative Analysis":
            model_selector = None
            training(
                model_type,
                model_selection_radio,
                model_selector,
                X_train,
                y_train,
                X_test,
                y_test,
                eval_metrics,
                comparison_metrics,
                slider,
            )

        if model_selection_radio == "Individual Model Selection":
            y_pred = training(
                model_type,
                model_selection_radio,
                model_selector,
                X_train,
                y_train,
                X_test,
                y_test,
                eval_metrics,
                comparison_metrics,
                slider,
            )

        if plot_generation_checkbox:
            if model_type == "Regression":
                if plot_selector == "Regression Plot":
                    st.markdown(
                        "<h2 style='text-align: center;'>Regression Plot</a></h2>",
                        unsafe_allow_html=True,
                    )
                    generate_regression_plot(y_test, y_pred)
            elif model_type == "Classification":
                if "Confusion Matrix" in plot_selector:
                    st.markdown(
                        "<h2 style='text-align: center;'>Confusion Matrix</a></h2>",
                        unsafe_allow_html=True,
                    )
                    generate_confusion_matrix(y_test, y_pred)
                if "ROC Curve" in plot_selector:
                    st.markdown(
                        "<h2 style='text-align: center;'>ROC Curve</a></h2>",
                        unsafe_allow_html=True,
                    )
                    generate_roc_curve(y_test, y_pred)

    else:
        st.sidebar.warning("Please upload a CSV file")
        fill_value = 0

    df.fillna(fill_value, inplace=True)

    target_column = st.sidebar.selectbox("Select the Target Column:", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.sidebar.header("Choose a Model")
    model_selector = st.sidebar.selectbox(
        "Select a model:",
        sorted(
            [
                name
                for name, _ in all_estimators(
                    type_filter=["regressor" if task == "Regression" else "classifier"]
                )
            ]
        ),
    )

    if st.sidebar.button("Submit"):
        submit_clicked = True

        selected_model = [
            est
            for name, est in all_estimators(
                type_filter=["regressor" if task == "Regression" else "classifier"]
            )
            if name == model_selector
        ][0]

        if task == "Regression":
            st.header("Regression Task")
            st.write(f"Selected Model: {model_selector}")

            model, mse, mae, r2 = perform_regression(
                X_train, y_train, X_test, y_test, selected_model
            )
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R-squared (R2): {r2}")

        elif task == "Classification":
            st.header("Classification Task")
            st.write(f"Selected Model: {model_selector}")

            model, accuracy, roc_auc, f1, recall = perform_classification(
                X_train, y_train, X_test, y_test, selected_model
            )
            st.write(f"Accuracy: {accuracy}")
            st.write(f"ROC AUC: {roc_auc}")
            st.write(f"F1 Score: {f1}")
            st.write(f"Recall: {recall}")


if __name__ == "__main__":
    main()
