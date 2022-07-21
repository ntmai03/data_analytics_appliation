import sys
import os
from pathlib import Path
import boto3
from io import BytesIO
import streamlit as st

import pandas as pd
import numpy as np
# split data
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
import sklearn
# to persist the model and the scaler
import joblib

from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf

import sklearn.decomposition as dec
import sklearn.datasets as ds
import sklearn.cluster as clu

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import csc_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.mixture import GaussianMixture

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# import seaborn to make nice plots
import seaborn as sns


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=False, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub('[^a-zA-Z\s]', '', text)
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

    

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

def plot_data(X, y, n_cluster):
    palette = np.array(sns.color_palette("hls", n_cluster))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
    sc = ax.scatter(X[:,0], X[:,1], s=3,c=y, cmap="jet")
    # We add the labels for each digit.
    txts = []
    for i in range(n_cluster):
        # Position of each label.
        xtext, ytext = np.median(X[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
                PathEffects.Stroke(linewidth=3, foreground="w"),
                PathEffects.Normal()])
        txts.append(txt)  
        
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)  
        

        
def compare_truelabel_cluster(X, y, cluster):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    ax[0].scatter(X[:,0], X[:,1], s=3, c=y, cmap='jet')
    ax[1].scatter(X[:,0], X[:,1], s=3, c=cluster, cmap='jet')
    plt.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)  


def visualize_transformed_data(x, y):    
    
    st.markdown('#### Visualize transformed data with TSNE')
    tsne = TSNE(n_components=2, random_state=42)
    tsne_value = tsne.fit_transform(x)
    plot_data(tsne_value, y, 10)


def show_missing_data(df):
    miss_val_df = pd.DataFrame(df.isnull().sum(), columns=['ColumnName'])
    miss_val_df['Percentage'] = 100 * df.isnull().sum()/len(df)
    miss_val_df.sort_values('Percentage', ascending=False)
    return miss_val_df