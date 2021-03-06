import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

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
        st.markdown('<p style="color:Green; font-size: 25px;"> 1.  Introduction</p>', unsafe_allow_html=True)
        st.write("The idea of topic modelling revolves around the process of extracting key themes or concepts from a corpus of documents which are represented as topics")
        st.write("There are varieties of techniques for topic modeling including both dimensionality reduction techniques and unsupervised techniques. This analysis will be going to employ some of these techniques to perform the task")
        st.write("The dataset used is “News category dataset” from Kaggle (https://www.kaggle.com/rmisra/news-category-dataset). This dataset is around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The original dataset contains over 30 categories, but for the purposes of this analysis, I will work with a subset of 3: TRAVEL, FOOD & DRINK, BUSINESS, SPORTS")
        
        st.markdown('<p style="color:Green; font-size: 25px;"> 2.  View Snapshot of dataset</p>', unsafe_allow_html=True)
        news = NewsCategory()
        news.load_ds()
        st.write('Full dataset dimensionality: ', news.data[news.RAW_VARS].shape)
        st.write(news.data[news.RAW_VARS].head())

        st.markdown('<p style="color:Green; font-size: 25px;"> 3.  Split data to train and test set</p>', unsafe_allow_html=True)
        st.write('Train set dimensionality: ', news.df_train[news.RAW_VARS].shape)
        st.write(news.df_train[news.RAW_VARS].head())
        st.write('Test set dimensionality: ', news.df_test[news.RAW_VARS].shape)
        st.write(news.df_test[news.RAW_VARS].head())



    #============================================= Data Processing ========================================
    if task_option == 'Data Processing':
        st.markdown('<p style="color:Green; font-size: 25px;"> 1. Text normalization</p>', unsafe_allow_html=True)
        news = NewsCategory()
        news.load_ds()
        st.write(news.data.head())
        st.markdown('<p style="color:Green; font-size: 25px;"> 2. Word Embedding Vectorizer</p>', unsafe_allow_html=True)
        news.preprocess_data(news.df_train, train_flag=1)



    #=========================================== Clustering ======================================
    if task_option == 'Auto Encoder':
        news = NewsCategory()
        st.markdown('<p style="color:Green; font-size: 25px;"> 1. Train auto encoder</p>', unsafe_allow_html=True)
        news.autoencoder_analysis(encoding1_dim=60, encoding2_dim=600, latent_dim=15)
        news.autoencoder_transform(news.X_train, 1)
        st.markdown('<p style="color:Green; font-size: 25px;"> 2. Embededed data from auto encoder</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Green; font-size: 25px;"> 3. Transform embededed data to tsne data for visualization</p>', unsafe_allow_html=True)
        news.transform_tsne(news.ae_embeddings, 1)
        news.tsne_data = news.load_preprocessed_ds('News_Category_tsne_data.csv')
        news.plot_cluster(news.tsne_data.values, news.y_train)


    if task_option == 'PCA':
        news = NewsCategory()
        news.pca_analysis(n_components=50)
        # news.transform_tsne(news.pca_data, 2)
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


        st.markdown('<p style="color:Green; font-size: 25px;"> 1. KMeans on raw data</p>', unsafe_allow_html=True)
        kmeans_label = news.kmeans_analysis(news.X_train, 4)
        cluster_df['KMeans'] = kmeans_label
        cluster_df['Cluster'] = np.nan
        cluster_df.loc[cluster_df.KMeans == 0,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 0].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 1,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 1].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 2,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 2].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 3,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 3].Class.mode()[0]
        news.tsne_data = news.load_preprocessed_ds('News_Category_tsne_data.csv')
        news.compare_cluster(news.tsne_data.values, news.y_train, cluster_df['Cluster'])


        st.markdown('<p style="color:Green; font-size: 25px;"> 2. PCA + KMeans</p>', unsafe_allow_html=True)
        news.pca_analysis(n_components=50)
        kmeans_pca = news.kmeans_analysis(news.pca_data, 4)
        cluster_df['KMeans'] = kmeans_pca 
        cluster_df['Cluster'] = np.nan
        cluster_df.loc[cluster_df.KMeans == 0,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 0].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 1,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 1].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 2,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 2].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 3,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 3].Class.mode()[0]
        news.compare_cluster(news.tsne_data.values, news.y_train, cluster_df['Cluster'])


        st.markdown('<p style="color:Green; font-size: 25px;"> 3. Auto Encoder + KMeans</p>', unsafe_allow_html=True)
        news.load_ae_embedding()
        kmeans_ae = news.kmeans_analysis(news.ae_embbeding, 4, 1)
        cluster_df['KMeans'] = kmeans_ae
        cluster_df['Cluster'] = np.nan
        cluster_df.loc[cluster_df.KMeans == 0,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 0].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 1,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 1].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 2,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 2].Class.mode()[0]
        cluster_df.loc[cluster_df.KMeans == 3,'Cluster'] = cluster_df.loc[cluster_df.KMeans == 3].Class.mode()[0]
        news.compare_cluster(news.tsne_data.values, news.y_train, cluster_df['Cluster'])


    
    if task_option == 'Prediction':
        news_input = st.text_input("Input text", 'new amusement park ride worth waiting line photosnewthissummer amusement park ride around world think worth wait mindnumbing line')
        news_input = usu.utils_preprocess_text(news_input, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
        news_object = pd.DataFrame(news_input, columns=['text_clean'])
        if st.button('Predict'):
            news = NewsCategory()
            news.data_preprocessing(news_object, train_flag=0)
            news.X_new = news.processed_data.drop([model.TARGET], axis=1)
            news.y_new = news.processed_data[model.TARGET]
            news.autoencoder_transform(news.X_new, 0)
            cluster = news.kmeans_analysis(news.ae_embbeding, 4, 2)
            st.write(cluster)
    


