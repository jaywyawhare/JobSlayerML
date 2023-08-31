import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

submit_clicked = False

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
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, mae, r2, y_pred

def perform_classification(X_train, y_train, X_test, y_test, selected_model):
    model = selected_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return model, accuracy, roc_auc, f1, recall, y_pred

def create_regression_plot(y_test, y_pred):
    st.subheader("Actual vs. Predicted Scatter Plot")
    trace_actual = go.Scatter(x=y_test, y=y_test, mode='markers', name='Actual', marker=dict(color='blue'))
    trace_predicted = go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted', marker=dict(color='red'))
    
    scatter_fig = go.Figure(data=[trace_actual, trace_predicted])
    scatter_fig.update_layout(
        xaxis_title='Actual Y',
        yaxis_title='Predicted Y',
        title='Actual vs. Predicted Values (Regression)'
    )
    st.plotly_chart(scatter_fig)

def create_classification_heatmap(y_test, y_pred):
    st.subheader("Classification Confusion Matrix Heatmap")
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    heatmap_fig = px.imshow(confusion_matrix, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(heatmap_fig)

def main():
    global submit_clicked
    global show_scatter_plot
    
    st.title("JobSlayerML : Who needs job security?")

    df = load_data()

    if df is None:
        return

    st.header("Data")
    st.write(df)

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
    
    show_scatter_plot = st.sidebar.checkbox("Show Scatter Plot", value=False)

    if st.sidebar.button("Submit"):
        submit_clicked = True 

        selected_model = [est for name, est in all_estimators(type_filter=['regressor' if task == 'Regression' else 'classifier']) if name == model_selector][0]

        if task == "Regression":
            st.header("Regression Task")
            st.write(f"Selected Model: {model_selector}")

            model, mse, mae, r2, y_pred = perform_regression(X_train, y_train, X_test, y_test, selected_model)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R-squared (R2): {r2}")
            create_regression_plot(y_test, y_pred)

        elif task == "Classification":
            st.header("Classification Task")
            st.write(f"Selected Model: {model_selector}")

            model, accuracy, roc_auc, f1, recall, y_pred = perform_classification(X_train, y_train, X_test, y_test, selected_model)
            st.write(f"Accuracy: {accuracy}")
            st.write(f"ROC AUC: {roc_auc}")
            st.write(f"F1 Score: {f1}")
            st.write(f"Recall: {recall}")
            create_classification_heatmap(y_test, y_pred)

if __name__ == "__main__":
    main()
