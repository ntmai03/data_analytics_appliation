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
from analysis.unsupervised import UnsupervisedAnalysis
from analysis.news_category import NewsCategory


# Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def app():

    st.sidebar.header('')
    st.sidebar.header('')
    data_type = ['Select Dataset',
                 'MNIST',
                 'News'
                 ]
    data_option = st.sidebar.selectbox('', data_type)
    st.sidebar.header('')

    st.sidebar.header('')
    st.sidebar.header('')
    task_type = ['Select Task',
                 'Dimension Reduction',
                 'Clustering',
                 'Anomaly Detection',
                 ]
    task_option = st.sidebar.selectbox('', task_type)
    st.sidebar.header('')


    #=========================================== Dimension Reduction ======================================
    if task_option == 'Dimension Reduction':
        reduction_type = ["Select Model",
                          "PCA", 
                          "SVD",
                          "AutoEncoder",
                          "SOM"]
        reduction_option = st.sidebar.selectbox('',reduction_type)
        st.sidebar.header('')
        st.sidebar.header('')

        # Load and show raw data
        if reduction_option == 'Select Model':
            st.markdown("#### Raw data:")
            if(data_option == 'MNIST'):
                model = UnsupervisedAnalysis()
                model.load_mnist_ds(flag=1)
                # Display the first 5 digits
                st.markdown('#### Display the first 5 digits of MNIST dataset')
                model.show_image()
            if(data_option == 'News'):
                model = NewsCategory()
                st.write('#### Display the first 5 rows')
                st.write(model.data.head())

        if reduction_option == 'PCA':
            if(data_option == 'MNIST'):
                model = UnsupervisedAnalysis()
            if(data_option == 'News'):
                model = NewsCategory()
            model.pca_analysis()
            model.pca_transform()   

        if reduction_option == 'SVD':
            if(data_option == 'MNIST'):
                model = UnsupervisedAnalysis()
            if(data_option == 'News'):
                st.write('TBC')
            model.svd_analysis()
            model.svd_transform()  

        if reduction_option == 'AutoEncoder':
            if(data_option == 'MNIST'):
                model = UnsupervisedAnalysis()
            if(data_option == 'News'):
                st.write('TBC')            
            model.autoencoder_analysis()
            model.autoencoder_transform()         


    #=========================================== Clustering ======================================
    if task_option == 'Clustering':
        st.sidebar.subheader('Clustering')
        cluster_type = ["Select Model",
                        "KMeans", 
                        "GMM",
                        "Hierarchical"]
        cluster_option = st.sidebar.selectbox('',cluster_type)
        st.sidebar.header('')

        if cluster_option == 'KMeans':
            if(data_option == 'MNIST'):
                model = UnsupervisedAnalysis()
            if(data_option == 'News'):
                st.write('TBC')             
            model.kmeans_analysis()
  
        if cluster_option == 'GMM':
            model = UnsupervisedAnalysis()
            model.gmm_analysis()



    #=========================================== Anomaly Detection ======================================    