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

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

# Modelling Helpers:
from sklearn.preprocessing import Normalizer, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, ShuffleSplit, cross_validate
from sklearn import model_selection
from sklearn.model_selection import train_test_split

import sklearn.decomposition as dec
from sklearn.manifold import TSNE
import sklearn.datasets as ds
import sklearn.cluster as clu
from sklearn.mixture import GaussianMixture

import statsmodels.api as sm
import sklearn

# to persist the model and the scaler
import joblib

from src.data_processing import diabetes_feature_engineering as fe
from src.util import data_manager as dm
from src import config as cf
# from src.pipeline import diabetes_pipeline as pl

# Evaluation metrics for Classification
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, 
                             r2_score, roc_auc_score, f1_score, precision_score, recall_score, 
                             precision_recall_curve, precision_recall_fscore_support, auc, 
                             average_precision_score)


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=6,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def scaler(data):
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    data_scaled.columns = data.columns
    
    return data


def relabel(cl):
    """ Relabel a clustering with three clusters to match the original classes"""
    if np.max(cl) != 2:
        return cl
    perms = np.array(list(permutations((0,1,2))))
    i = np.argmin([np.sum(np.abs(perm[1] - y)) for perm in perms])
    p = perms[i]
    return p[cl]



class UnsupervisedAnalysis:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    params={
    'changepoint_prior_scale':0.0018298282889708827,
    'holidays_prior_scale':0.00011949782374119523,
    'seasonality_mode':'additive',
    'seasonality_prior_scale':4.240162804451275
        }

    def __init__(self):
 
        self.TRAIN_VARS = None
        self.X = None
        self.y = None
        self.model = None
        self.model = None



    def load_digit_ds(self):
        df = ds.load_digits()
        self.X = pd.DataFrame(df.data)
        self.y = df.target
        self.images = df.images


    def show_image(self):
        nrows, ncols = 2, 5
        fig = plt.figure(figsize=(6,3))
        plt.gray()
        for i in range(ncols * nrows):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.matshow(self.images[i,...])
            plt.xticks([])
            plt.yticks([])
            plt.title(self.y[i])
        st.write(fig)    


    def pca_analysis(self, n_components=50, c=None, cmap='viridis'):

        pca = dec.PCA()
        pca_value = pca.fit_transform(self.X)
        
        # -------------plot variance ratio and cumulative sum--------------
        cumulative_list = []
        cumulative_sum = 0
        for v in pca.explained_variance_ratio_:
            cumulative_list.append(cumulative_sum + v)
            cumulative_sum = cumulative_list[-1]
            
        fig = plt.figure(figsize=(18,6))
        plt.subplot(1,2,1)
        plt.plot(pca.explained_variance_ratio_[0:n_components])
        features = list(range(0,n_components))
        plt.bar(features,pca.explained_variance_ratio_[0:n_components])
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        
        plt.subplot(1,2,2)
        plt.plot(cumulative_list[0:n_components])
        plt.xlabel('PCA features')
        plt.ylabel('cumulative sum')
        st.write(fig)
        
        
        # -----------------plot the first 2 components---------------------
        fig = plt.figure(figsize=(12,6))
        plt.scatter(pca_value[:,0], pca_value[:,1], s=5, c=c, alpha=0.5, cmap=cmap)
        # Add a legend
        # plt.legend(c,loc=1)
        plt.title('PCA Dimensionality Reduction')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Princiapl Component 2")
        st.write(fig)


    def svd_analysis(self, n_components=50, c=None, cmap='viridis'):
        svd = dec.TruncatedSVD()
        svd_value = svd.fit_transform(self.X)
        
        # plot the first 2 components
        fig = plt.figure(figsize=(12,6))
        plt.scatter(svd_value[:,0], svd_value[:,1], s=5, c=c, alpha=0.5, cmap=cmap)
        # Add a legend
        # plt.legend(c,loc=1)
        plt.title('SVD Dimensionality Reduction')
        plt.xlabel("SVD Component 1")
        plt.ylabel("SVD Component 2")
        st.write(fig)


    def tsne_analysis(self, c=None, cmap='viridis'):
        tsne = TSNE(n_components=2, random_state=9)
        tsne_value = tsne.fit_transform(self.X)
        
        # plot the first 2 components
        fig = plt.figure(figsize=(12,6))
        plt.scatter(tsne_value[:,0], tsne_value[:,1], s=5, c=c, 
                    alpha=0.5, cmap=cmap)
        # Add a legend
        # plt.legend(loc='best')
        plt.title('TSNE Dimensionality Reduction')
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        st.write(fig) 


    def clustering(self, n_clusters):
    
        pca = dec.PCA()
        pca_value = pca.fit_transform(self.X)
        PCA_df = pd.DataFrame(pca_value)
        selected_vars = range(0,15)
        PCA_df = PCA_df[selected_vars]

        fig, axes = plt.subplots(3, 3, figsize=(10,10), sharex=True, sharey=True)
        axes[0, 0].scatter(PCA_df.iloc[:,0], PCA_df.iloc[:,1], s=5, c=self.y, cmap=plt.cm.rainbow)
        axes[0, 0].set_title("True labels")

        for ax, est in zip(axes.flat[1:], [
            #clu.SpectralClustering(n_clusters),
            #clu.AgglomerativeClustering(n_clusters),
            #clu.MeanShift(),
            #clu.AffinityPropagation(),
            #clu.DBSCAN(),
            clu.KMeans(n_clusters),
            GaussianMixture(n_components=n_clusters, n_init=10),
        ]):
            # est.fit(X_data)
            c = relabel(est.fit_predict(self.X))
            ax.scatter(PCA_df.iloc[:,0], PCA_df.iloc[:,1], s=5, c=c, cmap=plt.cm.rainbow)
            ax.set_title(est.__class__.__name__)

          
        # Fix the spacing between subplots.
        fig.tight_layout() 
        st.write(fig)









