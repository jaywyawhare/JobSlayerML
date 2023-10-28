import streamlit as st
from inspect import signature
from sklearn.utils import all_estimators
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    roc_auc_score,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def models(model_type):
    """ 
    Function to return all the models for the specified model type.

    Args:
        model_type: The type of model to get. (Regression or Classification)

    Returns:
        A list of models for the specified model type.
    """
    if model_type == "Regression":
        regressorModels = all_estimators(type_filter="regressor")
        regressorModels.append(("XGBoost", XGBRegressor))
        regressorModels.append(("CatBoost", CatBoostRegressor))
        regressorModels.append(("LightGBM", LGBMRegressor))

        for name, estimator in regressorModels:
            if requiresPositionalArgument(estimator):
                regressorModels.remove((name, estimator))
        return regressorModels

    else:
        classifierModels = all_estimators(type_filter="classifier")
        classifierModels.append(("XGBoost", XGBClassifier))
        classifierModels.append(("CatBoost", CatBoostClassifier))
        classifierModels.append(("LightGBM", LGBMClassifier))

        for name, estimator in classifierModels:
            if requiresPositionalArgument(estimator):
                classifierModels.remove((name, estimator))
        return classifierModels


def eval(X_train, y_train, X_test, y_test, model, eval_metrics):
    """
    Function to evaluate the model and return the evaluation results.

    Args:
        X_train: The training data.
        y_train: The training target.
        X_test: The testing data.
        y_test: The testing target.
        model: The model to evaluate.
        eval_metrics: The evaluation metrics to use.

    Returns:
        The predictions from the model and the evaluation results.
    """
    try:
        if isinstance(model, type):
            model = model()
        else:
            model = model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        evaluation_results = {}

        metric_functions = {
            "Accuracy": accuracy_score,
            "Precision": precision_score,
            "Recall": recall_score,
            "F1": f1_score,
            "AUC": roc_auc_score,
            "MSE": mean_squared_error,
            "RMSE": lambda y_true, y_pred: mean_squared_error(
                y_true, y_pred, squared=False
            ),
            "MAE": mean_absolute_error,
            "R2": r2_score,
            "Adjusted R2": lambda y_true, y_pred: 1
            - (1 - r2_score(y_true, y_pred))
            * (len(y_true) - 1)
            / (len(y_true) - X_test.shape[1] - 1),
            "Explained Variance": explained_variance_score,
            "Max Error": max_error,
            "Mean Absolute Percentage Error": mean_absolute_percentage_error,
            "Median Absolute Error": median_absolute_error,
            "Mean Squared Log Error": mean_squared_log_error,
            "Mean Poisson Deviance": mean_poisson_deviance,
            "Mean Gamma Deviance": mean_gamma_deviance,
            "Mean Tweedie Deviance": mean_tweedie_deviance,
        }

        for metric in eval_metrics:
            if metric in metric_functions:
                score = metric_functions[metric](y_test, y_pred)
                evaluation_results[metric] = score
        return y_pred, evaluation_results
    except:
        pass


def featureExtraction(X_train, X_test, slider):
    """
    Function to perform PCA and return the top k components.

    Args:
        X_train: The training data.
        X_test: The testing data.
        slider: The number of components to use for PCA.

    Returns:
        The training and testing data with the top k components.
    """
    X_train.fillna(mean, inplace=True)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    pca = PCA(n_components=slider)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


def training(
    model_type,
    model_selection_radio,
    model_selector,
    X_train,
    y_train,
    X_test,
    y_test,
    eval_metrics,
    comparison_metrics,
    slider=None,
):
    """
    Function to train the either a single model or multiple models and display the results.

    Args:
        model_type : The type of model to train. (Regression or Classification)
        model_selection_radio: The type of model selection to use. (Comparative Analysis or Individual Model Selection)
        model_selector: The model to train. (Only used for Individual Model Selection)
        X_train: The training data. 
        y_train: The training target.
        X_test: The testing data.
        y_test: The testing target.
        eval_metrics: The evaluation metrics to use.
        comparison_metrics: The comparison metric to use.
        slider: The number of components to use for PCA.

    Returns:
        The predictions from the model.

    """
    if slider is not None:
        X_train, X_test = featureExtraction(X_train, X_test, slider)
    if model_selection_radio == "Comparative Analysis":
        st.header("Comparative Analysis")
        comparison_results = {}

        for name, model in models(model_type):
            try:
                y_pred, evaluation_results = eval(
                    X_train, y_train, X_test, y_test, model, eval_metrics
                )

                model_evaluation = {}
                for metrics, score in evaluation_results.items():
                    model_evaluation[metrics] = score

                comparison_results[name] = model_evaluation
            except:
                pass

        st.subheader("Best Model")
        best_model_name = max(
            comparison_results,
            key=lambda model_name: comparison_results[model_name][comparison_metrics],
        )
        st.write("Best Model:", best_model_name)
        st.subheader("Evaluation Results")
        results = pd.DataFrame(comparison_results)
        results = results.transpose()
        results = results.reset_index()
        results = results.rename(columns={"index": "Model"})
        st.table(results.style.set_properties(**{"text-align": "center"}))
        return 0

    else:
        st.header("Single Model")
        st.markdown(
            "<h2 style='text-align: center;'>Model Evaluation Results</a></h2>",
            unsafe_allow_html=True,
        )

        model = [est for name, est in models(model_type) if name == model_selector][0]

        y_pred, evaluation_results = eval(
            X_train, y_train, X_test, y_test, model, eval_metrics
        )
        best_model = pd.DataFrame({"Model": [model_selector], **evaluation_results})
        st.table(best_model.style.set_properties(**{"text-align": "center"}))
        return y_pred


def requiresPositionalArgument(model_class):
    """
    Check if a scikit-learn estimator class requires any positional arguments.

    Args:
        model_class: The scikit-learn estimator class.

    Returns:
        A list of required positional arguments, or an empty list if none are required.
    """
    required_args = []
    try:
        constructor = signature(model_class)
        parameters = constructor.parameters
        for arg_name, arg_info in parameters.items():
            if arg_info.default == arg_info.empty:
                required_args.append(arg_name)
    except Exception:
        pass
    return required_args
