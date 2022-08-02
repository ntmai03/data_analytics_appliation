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
                 'Data Understanding',
                 'Data Processing',
                 'PCA',
                 'Auto Encoder',
                 'Cluster Data',
                 'Prediction'
                 ]
    task_option = st.sidebar.selectbox('', task_type)
    st.sidebar.header('')


    #============================================= PART 1: Introduction ========================================
    if task_option == 'Introduction':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Problem and Objective</p>', unsafe_allow_html=True)
        st.write("**1. Problem**:The vast majority of the available data is actually unlabeled, so there is a huge potential in unsupervised learning that we can invest into. For instance, Topic modeling is one of the application of Unsupervised techniques. It is used to extract and depict key themes or topics which are latent in a large corpora of text documents.  ")
        st.write("**2. Objective**:This analysis combines techniques in NLP and Unsupervised learning to perform the task of topic modeling. These techniques are described in the following section")

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Techniques</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:green; font-size: 18px;"> 1. Text Preprocessing and Normalization</p>', unsafe_allow_html=True)
        st.write("Before feature engineering, we need to  pre-process, clean, and normalize text")
        st.markdown('<p style="color:green; font-size: 18px;"> 2. Word Embedding for Feature Extraction</p>', unsafe_allow_html=True)
        st.write("**1. Word Embedding technique**: For feature extraction, Word embedding is applied to transform each word to a numeric vector. Each word is assigned its own vector in such a way that words that frequently appear together in the same context are given vectors that are close together. So similar meaning words have similar representations, these vectors try to capture contextual and semantic information.")
        st.write("**2. Word2vec  Model with gensim**: This part employes the Word2vec model which was released by Google in 2013 and trained on the Google News dataset (about 100 billion words).  This step will transform each worrd to a numeric vector of 300 features and this high dimensionality dataset can expose some problems in Machine Learning model knnown as the curse of dimensionality.")
        st.write("**3. The curse of dimensionality**: Many Machine Learning algorithms have problems involving high dimensionality of features for each training instance. This makes it not only difficult to understand or explore the pattern of data, training time extremely slow, parameter estimation challenging, but also affects the machine learning model's performance since there are more chances to overfit the model or violate some of the assumptions of the algorithm. The problem is often referred to as the curse of dimensionality")
        st.write("")
        st.markdown('<p style="color:green; font-size: 18px;"> 3. Dimensionality Reduction with PCA and AutoEncoder</p>', unsafe_allow_html=True)
        st.write("**1. Dimensinoality Reduction with PCA and AutoEncoder**: This is where dimensionality reduction comes in. Dimensionality reduction is the process of reducing the number of random variables by obtaining a set of principal variables but still keep important information. The new variables are then used for downstream task such as Regression, Classification or Clustering")
        st.write("**2. Dimensionality Reduction with TSNE**: TSNE is a Dimensionality reduction technique which is extremely useful for data visualization. Reducing the number of dimensions down to two (or three) makes it possible to plot a condensed view of high-dimensional training set on a graph and often get better understand by visually detecing patterns, such as clusters.")
        st.markdown('<p style="color:green; font-size: 18px;"> 4. Clustering</p>', unsafe_allow_html=True)
        st.write("**1. Intuition**: Clustering is an unsupervesed learning method that divide the data points into a number of groups, such that the data points in the same groups have similar properties and data points in different groups have different properties in some sense. Unsupervised learning means that there is no outcome to be predicted, and the algorithm just tries to find patterns in the data. Clusters found may represent topics in text data")
        st.write("**2. Types of Clustering**: There is no univerrsal definition of what a cluster is: it really depends on the context, and different algorithms will capture different kinds of clusters. For example, some algorithm look for instances centered around a particular point, called centroid (KMeans, Gaussian Mixure). Others look for continous regions of densely packed instances: these clusters can take on any shape (DBScan). Some algorithms are hierarchical, looking for clusters of clusters (Agglomerative Clustering). ")    
        st.write("This analysis tried Clustering with original features, PCA features, and AutoEncoder features. The result is then compared to find the most appropriate dimensionality techniques ")

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Notebook</p>', unsafe_allow_html=True)
        st.write("More details and explanations at https://github.com/ntmai03/DataAnalysisProject/tree/main/03-Unsupervised")
        
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 4. Demo</p>', unsafe_allow_html=True)
        video_file = open('regression.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 5. Findings</p>', unsafe_allow_html=True)
        st.write("For text data, AutoEncoder outperforms PCA in compressing data in a way to represent pattern better")
        st.write("One of the good practice to recognize clusters in order to determine which algorithm is the most appropriate for a given dataset is trying to visualize data on a 2D space and looking for the pattern. The result is then verified by business users or experts of the field")


    #=========================================== PART 2: Data Understanding ====================================#
    if task_option == 'Data Understanding':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Dataset</p>', unsafe_allow_html=True)
        st.write("The dataset used is “News category dataset” from Kaggle (https://www.kaggle.com/rmisra/news-category-dataset). This dataset is around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The original dataset contains over 30 categories, but for the purposes of this analysis, I will work with a subset of 4: TRAVEL, FOOD & DRINK, BUSINESS, SPORTS")
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. View Snapshot of dataset</p>', unsafe_allow_html=True)
        news = NewsCategory()
        news.load_ds()
        st.write('Full dataset dimensionality: ', news.data[news.RAW_VARS].shape)
        st.write(news.data[news.RAW_VARS].head())
        st.write("**1. Create new column**: column 'text' is the concatenation of column 'headline' and 'short_description'. The analysis use column 'text' to cluster data")
        st.write("**2. View column text in the first five rows**:")
        for i in range(0, 5):
            st.write(news.data.text[i])
            st.write(" ")


    #============================================= PART 3: Data Processing ========================================
    if task_option == 'Data Processing':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1.  Overview</p>', unsafe_allow_html=True)
        st.write("**The following steps are applied to convert text data to numeric data**: ")
        st.write("1. Detect language and filter only English")
        st.write("2. Split data into train set and test set")
        st.write("3. Normalize text")
        st.write("4. Feature Extraction using word embedding with Word2Vec model")
        st.write("5. Scaling independent features")
        st.write("6. Convert target from categorical data to numeric data")

        news = NewsCategory()
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Split data to train and test set</p>', unsafe_allow_html=True)
        news.split_data()
        st.write('Train set dimensionality: ', news.df_train[news.RAW_VARS].shape)
        st.write(news.df_train[news.RAW_VARS].head())
        st.write('Test set dimensionality: ', news.df_test[news.RAW_VARS].shape)
        st.write(news.df_test[news.RAW_VARS].head())

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Text normalization</p>', unsafe_allow_html=True)
        st.write(news.df_train.head())

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 4. Word Embedding & Feature Scaling</p>', unsafe_allow_html=True)
        processed_data = news.preprocess_data(news.df_train, train_flag=1)
        st.write("Tasks finished")


    #=========================================== PART 4: Dimensionality Reduction ======================================
    if task_option == 'PCA':
        news = NewsCategory()
        if st.sidebar.button('Train model'): 
            news.pca_analysis(n_components=50)
            news.transform_tsne(news.pca_data, 2)
            tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_PCA_tsne_data.csv')
            news.plot_cluster(tsne_data.values, news.y_train)
        elif st.sidebar.button('Pre-trained model'):
            st.markdown('<p style="color:lightgreen; font-size: 25px;"> Transform PCA data to tsne data for visualization</p>', unsafe_allow_html=True)
            data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_preprocessed.csv') 
            y_train = data[news.TARGET] 
            tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_PCA_tsne_data.csv')
            news.plot_cluster(tsne_data.values, y_train)              



    if task_option == 'Auto Encoder':
        news = NewsCategory()
        if st.sidebar.button('Train model'): 
            st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Train auto encoder</p>', unsafe_allow_html=True)
            news.autoencoder_analysis(encoding1_dim=80, encoding2_dim=600, latent_dim=18)
            news.autoencoder_transform(news.X_train, 1)
            st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Transform embededed data to tsne data for visualization</p>', unsafe_allow_html=True)
            news.transform_tsne(news.ae_embeddings, 1)
            tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_tsne_data.csv')
            news.plot_cluster(tsne_data.values, news.y_train)
        elif st.sidebar.button('Pre-trained model'):
            st.markdown('<p style="color:lightgreen; font-size: 25px;"> Transform AutoEncoder data to tsne data for visualization</p>', unsafe_allow_html=True)
            data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_preprocessed.csv')
            y_train = data[news.TARGET]            
            tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_tsne_data.csv')
            news.plot_cluster(tsne_data.values, y_train)           



    #=========================================== PART 5: Clustering ======================================
    if task_option == 'Cluster Data':
        news = NewsCategory()
        news.load_ds()
        data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_preprocessed.csv')
        X_train = data.drop([news.TARGET], axis=1)
        y_train = data[news.TARGET]
        cluster_df = pd.DataFrame()
        cluster_df['Class'] = news.y_train


        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. KMeans on raw data</p>', unsafe_allow_html=True)
        kmeans_label = news.kmeans_analysis(X_train, 4)
        cluster_df['KMeans'] = kmeans_label
        cluster_class = news.mapping_cluster_class(y_train, kmeans_label)
        cluster_df['Cluster'] = cluster_df['KMeans'].map(cluster_class)
        tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_tsne_data.csv')
        news.compare_cluster(tsne_data.values, y_train, cluster_df['Cluster'])

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. PCA + KMeans</p>', unsafe_allow_html=True)
        news.pca_analysis(n_components=50)
        kmeans_pca = news.kmeans_analysis(news.pca_data, 4)
        cluster_df['KMeans'] = kmeans_pca 
        cluster_class = news.mapping_cluster_class(y_train, kmeans_pca)
        cluster_df['Cluster'] = cluster_df['KMeans'].map(cluster_class)
        tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_tsne_data.csv')
        news.compare_cluster(tsne_data.values, y_train, cluster_df['Cluster'])


        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Auto Encoder + KMeans</p>', unsafe_allow_html=True)
        ae_embbeding = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_ae_embbeding.csv')
        kmeans_ae = news.kmeans_analysis(ae_embbeding, 4, 1)
        cluster_df['KMeans'] = kmeans_ae
        cluster_class = news.mapping_cluster_class(y_train, kmeans_ae)
        cluster_df['Cluster'] = cluster_df['KMeans'].map(cluster_class)
        tsne_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'News_Category_tsne_data.csv')
        news.compare_cluster(tsne_data.values, y_train, cluster_df['Cluster'])

 

    #=========================================== PART 6: Prediction ======================================    
    if task_option == 'Prediction':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Input data</p>', unsafe_allow_html=True)
        news_input = st.text_input("Input text", 'U.S. Launches Auto Import Probe, China Vows To Defend Its Interests. The investigation could lead to new U.S. tariffs similar to those imposed on imported steel and aluminum in March')
        news_input = usu.utils_preprocess_text(news_input, flg_stemm=True, flg_lemm=True)
        news_object = dict({'text_clean': [news_input]})
        news_object = pd.DataFrame.from_dict(news_object)
        if st.button('Predict'):
            news = NewsCategory()
            st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Data Preprocessing</p>', unsafe_allow_html=True)
            processed_data = news.preprocess_data(news_object, train_flag=0)
            news.autoencoder_transform(processed_data, 0)
            for col in news.ae_embeddings.columns:
                news.ae_embeddings[col] = news.ae_embeddings[col].astype(float)
            cluster = news.kmeans_analysis(news.ae_embeddings, 4, 2)
            predicted_category = list(news.TARGET_VALUE.keys())[2]
            st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Predict category</p>', unsafe_allow_html=True)
            st.write(predicted_category)
    


