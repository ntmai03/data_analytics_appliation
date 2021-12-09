# Python ≥3.5 is required
import sys
from pathlib import Path
import os

import streamlit as st

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.patheffects as PathEffects

from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf


hotel_stacked_ae_file = os.path.join(cf.TRAINED_MODEL_PATH, 'hotel_stacked_ae.pkl')



class HotelRecommendation:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    ##############################################################################################
    # Define parameters
    ##############################################################################################

                  

    ##############################################################################################
    # Initialize class oject
    ##############################################################################################
    def __init__(self):
 
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.pred_train = None
        self.pred_test = None
        self.prob_train = None
        self.prob_test = None
        self.processed_X_train = None
        self.processed_X_test = None


    ##############################################################################################
    # Data Processing
    ##############################################################################################
    def load_dataset(self):
        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "airbnb_train_corpus.csv")



    def data_processing_pipeline(self, df, train_flag=0):

        df = self.clean_data(df)
        df = self.create_season(df, self.TEMPORAL_VARS)
        df = self.create_sqft_ratio(df, 'sqft_living', 'sqft_living15')
        df = self.encode_categorical_ordinal(df, self.TEMP_CATEGORICAL_VARS, self.TARGET, train_flag)
        df = self.impute_na_median(df, self.NUMERICAL_VARS_WITH_NA, train_flag)

        data_scaled = self.scaling_data(df, self.NUMERICAL_VARS, train_flag)
        data_categorical = self.create_dummy_vars(df, self.CATEGORICAL_VARS, train_flag)
        df = pd.concat([data_scaled,data_categorical], axis=1)

        return df



    ##############################################################################################
    # Dimensionality Reduction
    ##############################################################################################
    def train_stacked_ae_data(self):

        #stacked_ae_corpus =  dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'stacked_ae_train_corpus.csv')
        stacked_ae_tsne = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'stacked_ae_tsne.csv')
        stacked_ae = joblib.load(hotel_stacked_ae_file)
        # show tsne
        plt.figure(figsize=(10,5))
        plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], s=5)
        plt.show()


    ##############################################################################################
    # Topic Modeling
    ##############################################################################################
    def train_kmeans(self):

        stacked_ae_tsne = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'stacked_ae_tsne.csv')
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=9, init='k-means++')
            kmeans.fit(X_train_ae)
            wcss.append(kmeans.inertia_)
            
        plt.plot(range(1,11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('K Clusters')
        plt.ylabel('WCSS (Error)')
        plt.show()

        model, cluster_df = train_kmeans(X_train_ae, ncluster = 6)
        # calculate mean and std
        cluster_df.head(10)


    def plot_cluster(X, y):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
        ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="jet",s=3)
        #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="jet"), ax=ax[0])
        ax[1].scatter(X[:, 0], X[:, 1], s=3)
        #plt.colorbar()
        plt.axis("off")
        plt.show()  


    

    def scatter(x, colors, n_cluster):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", n_cluster))

        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=6,c=palette[colors.astype(np.int)], cmap="jet")
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=6,c=colors, cmap="jet")
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(n_cluster):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        return f, ax, sc, txts

    
    def show_cluster(self):
        df_train['Cluster'] = cluster_df['Cluster']
        df_train.head()


        

