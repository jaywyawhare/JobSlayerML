import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.figure_factory as ff


def generate_regression_plots(X_train, X_test, y_train, y_test, model):
    fig = ff.create_regression_plot(model, X_train, y_train, kind="reg")
    st.write("Regression Plot")
    st.plotly_chart(fig)

def generate_confusion_matrix(y_true, y_pred):
    st.header("Classification Results")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True)
    st.write(cm)    

def generate_heatmap(corr_matrix):
    st.subheader("Correlation Heatmap")
    sns.heatmap(corr_matrix, annot=True)
    st.pyplot()

def generate_scatter_matrix(df):
    st.subheader("Scatter Matrix")
    sns.pairplot(df)
    st.pyplot()

def generate_boxplot(df):
    st.subheader("Box Plot")
    sns.boxplot(data=df)
    st.pyplot()

def generate_histogram(df):
    st.subheader("Histogram")
    sns.histplot(data=df)
    st.pyplot()

def generate_barplot(df):
    st.subheader("Bar Plot")
    sns.barplot(data=df)
    st.pyplot()

def generate_countplot(df):
    st.subheader("Count Plot")
    sns.countplot(data=df)
    st.pyplot()

def generate_violinplot(df):
    st.subheader("Violin Plot")
    sns.violinplot(data=df)
    st.pyplot()
