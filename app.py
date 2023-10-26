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
        if fillna_checkbox:
            fillna_option = st.sidebar.selectbox(
                "Select the fillna option", ["Mean", "Mode", "0"]
            )

        exploratory_data_analysis_checkbox = st.sidebar.checkbox(
            "Exploratory Data Analysis"
        )
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
            encoding_columns = st.sidebar.multiselect(
                "Select the columns to encode", df.columns
            )

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
        if train_test_split_checkbox:
            test_size = st.sidebar.slider("Select the test size", 0.0, 1.0, 0.2, 0.05)
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

        if not submit_clicked:
            return None

        if drop_columns_checkbox:
            df = drop_columns(df, columns_to_drop)
        if fillna_checkbox:
            df = fill_nan(df, fillna_option)
        if encode_checkbox:
            df = encode_categorical(df, encoding_option, encoding_columns)
        if scale_checkbox:
            df, scaler = data_scaling(df, scaling_option, scaling_columns)
        if train_test_split_checkbox:
            X_train, X_test, y_train, y_test = train_test_validation_split(
                df, target_column, test_size, random_state
            )

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


if __name__ == "__main__":
    main()
