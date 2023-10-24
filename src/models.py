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
    fbeta_score,
    roc_auc_score,
    roc_curve,
)


def models(model_type):
    if model_type == "Regression":
        regressorModels = all_estimators(type_filter="regressor")
        regressorModels.append(("XGBoost", XGBRegressor()))
        regressorModels.append(("CatBoost", CatBoostRegressor()))
        regressorModels.append(("LightGBM", LGBMRegressor()))
        return regressorModels
    else:
        classifierModels = all_estimators(type_filter="classifier")
        classifierModels.append(("XGBoost", XGBClassifier()))
        classifierModels.append(("CatBoost", CatBoostClassifier()))
        classifierModels.append(("LightGBM", LGBMClassifier()))
        return classifierModels


def eval(X_train, y_train, X_test, y_test, model, eval_metrics):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluation_results = {}

    metric_functions = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1_score,
        "F2": fbeta_score,
        "AUC": roc_auc_score,
        "ROC": roc_curve,
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

    return evaluation_results


def training(
    model_type,
    model_selector,
    X_train,
    y_train,
    X_test,
    y_test,
    eval_metrics,
    comparison_metrics,
):
    if model_selector == "Comparative Analysis":
        st.header("Comparative Analysis")
        comparison_DF = pd.DataFrame(columns=["Model"] + eval_metrics)
        for name, model in models(model_type):
            evaluation_results = eval(
                X_train, y_train, X_test, y_test, model, eval_metrics
            )
            comparison_DF = pd.concat(
                [comparison_DF, pd.DataFrame({"Model": [name], **evaluation_results})],
                axis=0,
            )
        comparison_DF = comparison_DF.reset_index(drop=True)
        st.subheader("Best Model")
        # sort by the comparison metric and show sorted dataframe
        st.write(comparison_DF.sort_values(by=comparison_metrics, ascending=False))
        best_model = comparison_DF.sort_values(
            by=comparison_metrics, ascending=False
        ).iloc[0]
        st.write(best_model)

    else:
        st.header("Single Model")
        model = [est for name, est in models(model_type) if name == model_selector][0]
        evaluation_results = eval(X_train, y_train, X_test, y_test, model, eval_metrics)
        st.write(evaluation_results)
        best_model = pd.DataFrame({"Model": [model_selector], **evaluation_results})
        st.subheader("Best Model")
        st.write(best_model)
        return y_test, y_pred, best_model
