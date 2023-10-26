import pandas as pd
import streamlit as st
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, auc, roc_curve


def generate_heatmap(df):
    """
    Function to generate a heatmap for the correlation matrix of the dataframe.

    Args:
        df: The dataframe to generate the heatmap for.

    Returns:
        None
    """
    corr = df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        showscale=True,
    )

    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig)


def generate_distribution_plot(df):
    """
    Function to generate a distribution plot for the dataframe.

    Args:
        df: The dataframe to generate the distribution plot for.

    Returns:
        None
    """
    numeric_cols = [col for col in df.columns if df[col].dtype != "object"]
    n = len(numeric_cols) // 4 + 1
    fig = make_subplots(rows=4, cols=n)

    for i, col in enumerate(numeric_cols):
        histogram = go.Histogram(x=df[col], name=col)
        fig.add_trace(histogram, row=i // n + 1, col=i % n + 1)

    fig.update_layout(height=1000, width=1500)
    st.plotly_chart(fig)


def generate_pca_plot(df, target_column):
    """
    Function to generate a PCA plot, with the target column. (Cummulative Variance Explained vs Number of Components)

    Args:
        df: The dataframe to generate the PCA plot for.
        target_column: The target column.   

    Returns:
        None
    """
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(df_std)

    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance_ratio) + 1)),
            y=cumulative_variance_ratio,
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="PCA - Cumulative Variance Explained",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained",
    )

    st.plotly_chart(fig, use_container_width=True)


def generate_2d_tsne(df, target_column):
    """
    Function to generate a 2D TSNE plot, with the target column.

    Args:
        df: The dataframe to generate the 2D TSNE plot for.
        target_column: The target column.

    Returns:
        None
    """
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(df_std)

    df_tsne = pd.DataFrame(X_tsne, columns=["x", "y"])
    df_tsne["target"] = df[target_column].values
    fig = px.scatter(
        df_tsne, x="x", y="y", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="2D TSNE")
    st.plotly_chart(fig, use_container_width=True)


def generate_2d_umap(df, target_column):
    """
    Function to generate a 2D UMAP plot, with the target column.

    Args:
        df: The dataframe to generate the 2D UMAP plot for.
        target_column: The target column.

    Returns:
        None
    """
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)

    umap_ = umap.UMAP(n_components=2)
    X_umap = umap_.fit_transform(df_std)

    df_umap = pd.DataFrame(X_umap, columns=["x", "y"])
    df_umap["target"] = df[target_column].values
    fig = px.scatter(
        df_umap, x="x", y="y", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="2D UMAP")
    st.plotly_chart(fig, use_container_width=True)


def generate_3d_tsne(df, target_column):
    """
    Function to generate a 3D TSNE plot, with the target column.

    Args:
        df: The dataframe to generate the 3D TSNE plot for.
        target_column: The target column.

    Returns:
        None
    """
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)

    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(df_std)

    df_tsne = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
    df_tsne["target"] = df[target_column].values
    fig = px.scatter_3d(
        df_tsne, x="x", y="y", z="z", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="3D TSNE")
    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig, use_container_width=True)


def generate_3d_umap(df, target_column):
    """
    Function to generate a 3D UMAP plot, with the target column.

    Args:
        df: The dataframe to generate the 3D UMAP plot for.
        target_column: The target column.

    Returns:
        None
    """
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)

    umap_ = umap.UMAP(n_components=3)
    X_umap = umap_.fit_transform(df_std)

    df_umap = pd.DataFrame(X_umap, columns=["x", "y", "z"])
    df_umap["target"] = df[target_column].values
    fig = px.scatter_3d(
        df_umap, x="x", y="y", z="z", color="target", color_continuous_scale="Blues"
    )
    fig.update_layout(title="3D UMAP")
    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig, use_container_width=True)


def generate_regression_plot(y_test, y_pred):
    """
    Function to generate a regression plot.

    Args:
        y_test: The testing target.
        y_pred: The predictions from the model.

    Returns:
        None
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
        )
    )
    fig.update_layout(
        title="Regression Plot",
        xaxis_title="Actual",
        yaxis_title="Predicted",
    )

    st.plotly_chart(fig, use_container_width=True)


def generate_confusion_matrix(y_test, y_pred):
    """
    Function to generate a confusion matrix.

    Args:
        y_test: The testing target.
        y_pred: The predictions from the model.

    Returns:
        None
    """
    fig = go.Figure()
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(y_test.unique()),
        y=list(y_test.unique()),
        annotation_text=cm.round(2),
        showscale=True,
    )

    fig.update_layout(width=1000, height=1000)
    st.plotly_chart(fig)


def generate_roc_curve(y_test, y_pred):
    """
    Function to generate a ROC curve.

    Args:
        y_test: The testing target.
        y_pred: The predictions from the model.

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name="ROC curve (area = %0.2f)" % roc_auc
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False
        )
    )

    fig.update_layout(
        title="Receiver operating characteristic (ROC) curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0, y=1, bordercolor="black", borderwidth=1),
        margin=dict(l=50, r=50, t=50, b=50),
        width=800,
        height=600,
    )

    st.plotly_chart(fig)
