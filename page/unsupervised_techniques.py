import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

from scipy import stats

# Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from src import config as cf
from src.util import data_manager as dm
from src.analysis.news_category import NewsCategory
from src.util import unsupervised_util as usu


def app():

    st.sidebar.header('')
    st.sidebar.subheader('Select function')
    task_type = ['Introduction',
                 'Data Processing',
                 'Auto Encoder',
                 'PCA',
                 'Cluster Data',
                 'Prediction'
                 ]
    task_option = st.sidebar.selectbox('', task_type)
    st.sidebar.header('')


    #============================================= Introduction ========================================
    if task_option == 'Introduction':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1.  Introduction</p>', unsafe_allow_html=True)
        st.write("The idea of topic modelling revolves around the process of extracting key themes or concepts from a corpus of documents which are represented as topics")
        st.write("There are varieties of techniques for topic modeling including both dimensionality reduction techniques and unsupervised techniques. This analysis will be going to employ some of these techniques to perform the task")
        st.write("The dataset used is “News category dataset” from Kaggle (https://www.kaggle.com/rmisra/news-category-dataset). This dataset is around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The original dataset contains over 30 categories, but for the purposes of this analysis, I will work with a subset of 3: TRAVEL, FOOD & DRINK, BUSINESS, SPORTS")

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Notebook</p>', unsafe_allow_html=True)
        st.write("More details and explanations at https://github.com/ntmai03/DataAnalysisProject/tree/main/03-Unsupervised")
        
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. View Snapshot of dataset</p>', unsafe_allow_html=True)
        news = NewsCategory()
        news.load_ds()
        st.write('Full dataset dimensionality: ', news.data[news.RAW_VARS].shape)
        st.write(news.data[news.RAW_VARS].head())
        st.write("Column 'text' is the concatenation of column 'headline' and 'short_description'. The analysis use column 'text' to cluster data")
        st.write("View column text in the first five rows:")
        for i in range(0, 5):
            st.write(news.data.text[i])
            st.write(" ")


    #============================================= Data Processing ========================================
    if task_option == 'Data Processing':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1.  Overview</p>', unsafe_allow_html=True)
        st.write("The following steps are applied to convert text data to numeric data: ")
        st.write("Detect language and filter only English")
        st.write("Split data into train set and test set")
        st.write("Normalize text")
        st.write("Feature Extraction using word embedding with Word2Vec model")
        st.write("Scaling independent features")
        st.write("Convert target from categorical data to numeric data")

        news = NewsCategory()
        news.load_preprocessed_ds("normalized_News_Category.csv")
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Split data to train and test set</p>', unsafe_allow_html=True)
        news.split_data()
        st.write('Train set dimensionality: ', news.df_train[news.RAW_VARS].shape)
        st.write(news.df_train[news.RAW_VARS].head())
        st.write('Test set dimensionality: ', news.df_test[news.RAW_VARS].shape)
        st.write(news.df_test[news.RAW_VARS].head())

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Text normalization</p>', unsafe_allow_html=True)
        st.write(news.df_train.head())

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 4. Word Embedding Vectorizer</p>', unsafe_allow_html=True)
        processed_data = news.preprocess_data(news.df_train, train_flag=1)
        processed_data.head()


    #=========================================== Clustering ======================================
    if task_option == 'Auto Encoder':
        news = NewsCategory()
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Train auto encoder</p>', unsafe_allow_html=True)
        news.autoencoder_analysis(encoding1_dim=80, encoding2_dim=600, latent_dim=18)
        news.autoencoder_transform(news.X_train, 1)
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Transform embededed data to tsne data for visualization</p>', unsafe_allow_html=True)
        news.transform_tsne(news.ae_embeddings, 1)
        news.tsne_data = news.load_preprocessed_ds('News_Category_tsne_data.csv')
        news.plot_cluster(news.tsne_data.values, news.y_train)


    if task_option == 'PCA':
        news = NewsCategory()
        news.pca_analysis(n_components=50)
        news.transform_tsne(news.pca_data, 2)
        news.tsne_data = news.load_preprocessed_ds('News_Category_PCA_tsne_data.csv')
        news.plot_cluster(news.tsne_data.values, news.y_train)


    if task_option == 'Cluster Data':
        news = NewsCategory()
        news.load_ds()
        data = news.load_preprocessed_ds('News_Category_preprocessed.csv')
        news.X_train = data.drop([news.TARGET], axis=1)
        news.y_train = data[news.TARGET]
        cluster_df = pd.DataFrame()
        cluster_df['Class'] = news.y_train


        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. KMeans on raw data</p>', unsafe_allow_html=True)
        kmeans_label = news.kmeans_analysis(news.X_train, 4)
        cluster_df['KMeans'] = kmeans_label
        cluster_class = news.mapping_cluster_class(news.y_train, kmeans_label)
        cluster_df['Cluster'] = cluster_df['KMeans'].map(cluster_class)
        news.tsne_data = news.load_preprocessed_ds('News_Category_tsne_data.csv')
        news.compare_cluster(news.tsne_data.values, news.y_train, cluster_df['Cluster'])


        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. PCA + KMeans</p>', unsafe_allow_html=True)
        news.pca_analysis(n_components=50)
        kmeans_pca = news.kmeans_analysis(news.pca_data, 4)
        cluster_df['KMeans'] = kmeans_pca 
        cluster_class = news.mapping_cluster_class(news.y_train, kmeans_pca)
        cluster_df['Cluster'] = cluster_df['KMeans'].map(cluster_class)
        news.tsne_data = news.load_preprocessed_ds('News_Category_tsne_data.csv')
        news.compare_cluster(news.tsne_data.values, news.y_train, cluster_df['Cluster'])


        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Auto Encoder + KMeans</p>', unsafe_allow_html=True)
        news.load_ae_embedding()
        kmeans_ae = news.kmeans_analysis(news.ae_embbeding, 4, 1)
        cluster_df['KMeans'] = kmeans_ae
        cluster_class = news.mapping_cluster_class(news.y_train, kmeans_ae)
        cluster_df['Cluster'] = cluster_df['KMeans'].map(cluster_class)
        news.tsne_data = news.load_preprocessed_ds('News_Category_tsne_data.csv')
        news.compare_cluster(news.tsne_data.values, news.y_train, cluster_df['Cluster'])

    
    if task_option == 'Prediction':
        news_input = st.text_input("Input text", 'Hotels That Take You Back in Time. These ancient dwellings are of high historical, artistic and architectural value; they are places not only to visit, but also to live and experience.')
        news_input = usu.utils_preprocess_text(news_input, flg_stemm=True, flg_lemm=True)
        news_object = dict({'text_clean': [news_input]})
        news_object = pd.DataFrame.from_dict(news_object)
        if st.button('Predict'):
            news = NewsCategory()
            processed_data = news.preprocess_data(news_object, train_flag=0)
            news.autoencoder_transform(processed_data, 0)
            processed_data = float(processed_data)
            cluster = news.kmeans_analysis(news.ae_embeddings, 4, 2)
            st.write(cluster)
    


