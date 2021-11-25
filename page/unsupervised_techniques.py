import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

sys.path.append('src')
from src import config as cf
from src.util import data_manager as dm
from src.data_processing import diabetes_feature_engineering as fe
from analysis.unsupervised import UnsupervisedAnalysis


# Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def app():

    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.subheader('Dimension Reduction')
    reduction_type = ["Select Model",
                      "PCA", 
                      "SVD",
                      "AutoEncoder",
                      "Variational AutoEncoder",
                      "TSNE",
                      "LDA",
                      "RBM",
                      "SOM",
                      "Probablistic Non MatrixFactorization"]
    reduction_option = st.sidebar.selectbox('',reduction_type)
    st.sidebar.header('')
    st.sidebar.header('')


    st.sidebar.subheader('Clustering')
    cluster_type = ["Select Model",
                    "KMeans", 
                    "Hierarchy",
                    "DBScan",
                    "Affinity",
                    "GMM"]
    cluster_option = st.sidebar.selectbox('',cluster_type)
    st.sidebar.header('')
    st.sidebar.header('')

    model = UnsupervisedAnalysis()
    model.load_digit_ds()
    model.show_image()


    if reduction_option == 'PCA':
        model = UnsupervisedAnalysis()
        model.load_digit_ds()     
        model.pca_analysis(c=model.y)  

    if reduction_option == 'SVD':
        model = UnsupervisedAnalysis()
        model.load_digit_ds()     
        model.svd_analysis(c=model.y)  

    if reduction_option == 'TSNE':
        model = UnsupervisedAnalysis()
        model.load_digit_ds()     
        model.tsne_analysis(c=model.y)  

    if cluster_option == 'Select Model':
        model = UnsupervisedAnalysis()
        model.load_digit_ds()     
        model.clustering(n_clusters=10)          