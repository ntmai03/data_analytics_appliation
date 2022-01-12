# Python ≥3.5 is required
import sys
from pathlib import Path
import os
from io import BytesIO
import datetime

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

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

import statsmodels.api as sm
import sklearn

# to persist the model and the scaler
import joblib

from src.data_processing import diabetes_feature_engineering as fe
from src.util import data_manager as dm
from src.util import unsupervised_util as unu
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

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Deep Learnign libraries
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

class UnsupervisedAnalysis:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """

    def __init__(self):
 
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None


    def load_mnist_ds(self, flag=0):
        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_valid, X_train = X_train_full[:5000], X_train_full[45000:]
        y_valid, y_train = y_train_full[:5000], y_train_full[45000:]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid

        if(flag == 1):
            st.write(X_train.shape)
            st.write(X_train[0:2])


    def show_image(self, n_images=5):
        nrows, ncols = 1, 5
        fig = plt.figure(figsize=(n_images * 1.5, 2))
        plt.gray()
        for i in range(ncols * nrows):
            ax = plt.subplot(nrows, ncols, i + 1)
            #ax.imshow(self.X_train[0:n_images].reshape(len(self.X_train[0:n_images]),28,28)[i])
            ax.imshow(self.X_train[i])
            plt.xticks([])
            plt.yticks([])

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def process_data(self):
        self.scaled_Xtrain = self.X_train / 255.
        self.scaled_Xtest = self.X_test / 255.
        self.scaled_Xvalid = self.X_valid / 255.

        self.scaled_Xtrain = self.scaled_Xtrain.reshape(-1, 784)
        self.scaled_Xtest = self.scaled_Xtest.reshape(-1, 784)
        self.scaled_Xvalid = self.scaled_Xtest.reshape(-1, 784)


    def pca_analysis(self):

        pca = PCA()
        self.process_data()
        X_train_pca = pca.fit(self.scaled_Xtrain)
        
        # -------------plot variance ratio and cumulative sum--------------
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1
        st.write()
        st.markdown('#### Number of principal components corresponding to 95 percent of data: ')
        st.write(str(d))
         
        st.markdown('#### Select number of principal components')   
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        ax[0].plot(pca.explained_variance_ratio_[0:500])
        ax[0].grid(True)

        ax[1].plot(cumsum, linewidth=3)
        ax[1].axis([0, 400, 0, 1])
        plt.xlabel("Dimensions")
        plt.ylabel("Cumsum Explained Variance")
        ax[1].plot([d, d], [0, 0.95], "k:")
        ax[1].plot([0, d], [0.95, 0.95], "k:")
        ax[1].plot(d, 0.95, "ko")
        ax[1].annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7), arrowprops=dict(arrowstyle="->"), fontsize=16)
        plt.grid(True)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def pca_transform(self, n_components=0.80):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(self.scaled_Xtrain)

        st.markdown("#### Number of principal components corresponding to 80 percent of variance")
        st.write(pca.n_components_, np.sum(pca.explained_variance_ratio_))
        st.markdown("#### PCA transformed data")
        st.write(X_train_pca[0:3])
        unu.visualize_transformed_data(X_train_pca, self.y_train)


    def svd_analysis(self):
        self.process_data()
        U, S, V = np.linalg.svd(self.scaled_Xtrain)

        st.markdown('#### The 10 largest values:' )
        st.write(S[0:10])
        self.U = U
        self.S = S
        self.V = V

        # Select number of latent vars
        st.markdown('#### Select number of latent vars')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        ax.plot(S, 'ks-')
        plt.xlabel('Component number')
        plt.ylabel('$\sigma$')
        plt.title('"Scree plot" of singular values')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def svd_transform(self, n_components=30):
        # list the components you want to use for the reconstruction
        n_comps = np.arange(0,n_components)

        # reconstruct the low-rank version of the picture
        X_pred = self.U[:,n_comps]@np.diag(self.S[n_comps])@self.V[:,n_comps].T 
        
        st.markdown("#### SVD Transformed data")
        st.write(self.U[:5,n_comps])
        unu.visualize_transformed_data(self.U[:,n_comps], self.y_train)               


    def autoencoder_analysis(self):

        self.process_data()

        # Construct architecture
        tf.random.set_seed(42)
        np.random.seed(42)
        keras.backend.clear_session()
        tf.keras.backend.clear_session()

        # No of Neurons in each layer []
        input_dim = self.scaled_Xtrain.shape[1]
        encoding1_dim = 100
        encoding2_dim = 550
        latent_dim = 20

        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(int(encoding1_dim), activation="relu")(input_layer)
        encoder = Dense(int(encoding2_dim), activation='relu')(encoder)
        encoder = Dense(int(latent_dim), activation='relu')(encoder)
        decoder = Dense(int(encoding2_dim), activation='relu')(encoder)
        decoder = Dense(int(encoding1_dim), activation='relu')(decoder)
        decoder = Dense(int(input_dim), activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)

        # this models maps an input to its encoded representation
        encoder_layer = Model(input_layer, encoder)
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(int(latent_dim),))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-3]
        # create the decoder model
        decoder_layer = Model(encoded_input, decoder_layer(encoded_input)) 

        # train model
        nb_epoch = 20
        batch_size = 10
        autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder = autoencoder
        self.encoder_layer = encoder_layer

        cp = ModelCheckpoint(filepath='autoencoder1.h5', save_best_only=True, verbose=0)
        tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        t_ini = datetime.datetime.now()
        history = autoencoder.fit(self.scaled_Xtrain, 
                                  self.scaled_Xtrain, 
                                  epochs=nb_epoch, 
                                  #batch_size=batch_size, 
                                  shuffle=True, 
                                  validation_data=(self.scaled_Xvalid, self.scaled_Xvalid),
                                  #verbose=1,
        ).history
        t_fin = datetime.datetime.now()
        st.write('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))

        df_history = pd.DataFrame(history) 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        ax.plot(df_history['loss'], linewidth=2, label='Train')
        ax.plot(df_history['val_loss'], linewidth=2, label='Test')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def autoencoder_transform(self, n_components=30):
        X_train_ae = self.encoder_layer(self.scaled_Xtrain)

        st.markdown("#### AutoeEncoder Transformed data")
        st.write(X_train_ae[:5])
        unu.visualize_transformed_data(X_train_ae, self.y_train)   


    def kmeans_analysis(self, n_clusters=10):


        # get data from s3
        X_train_ae = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'Mnist_ae.csv')
        X_train_ae_tsne = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'Mnist_ae_tsne.csv')
     
        kmeans = KMeans(n_clusters=n_clusters, random_state=9, init='k-means++')
        kmeans_cluster = kmeans.fit_predict(X_train_ae)

        # visualize cluster
        unu.compare_truelabel_cluster(X_train_ae_tsne.values, self.y_train, kmeans_cluster)

        # specifyign number of clusters with Elbow method
        kmeans_per_k = [KMeans(n_clusters=k, random_state=9).fit(X_train_ae) for k in range(1, 20)]
        inertias = [model.inertia_ for model in kmeans_per_k]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4))
        ax.plot(range(1, 20), inertias, "bo-")
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Inertia", fontsize=14)
        plt.title('Elbow Method')
        plt.annotate('Elbow',
                     xy=(10, inertias[9]),
                     xytext=(0.55, 0.55),
                     textcoords='figure fraction',
                     fontsize=16,
                     arrowprops=dict(facecolor='black', shrink=0.1)
                    )
        ax.axis([1, 20, 300000, 700000])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf) 


    def gmm_analysis(self, n_clusters=10):

        # get data from s3
        X_train_ae = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'Mnist_ae.csv')
        X_train_ae_tsne = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'Mnist_ae_tsne.csv')
      
        gmm = GaussianMixture(n_components=10, n_init=10, random_state=9)
        gmm.fit(X_train_ae)

        # visualize cluster
        unu.compare_truelabel_cluster(X_train_ae_tsne.values, self.y_train, gmm.predict(X_train_ae))

        # specifyign number of clusters with Elbow method
        gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=9).fit(X_train_ae)
             for k in range(1, 20)]

        bics = [model.bic(X_train_ae) for model in gms_per_k]
        aics = [model.aic(X_train_ae) for model in gms_per_k]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4))
        ax.plot(range(1, 20), bics, "bo-", label="BIC")
        ax.plot(range(1, 20), aics, "go--", label="AIC")
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Information Criterion", fontsize=14)
        plt.axis([1, 19.5, np.min(aics) - 50, np.max(aics) + 50])
        plt.annotate('Minimum',
                     xy=(10, bics[9]),
                     xytext=(0.35, 0.6),
                     textcoords='figure fraction',
                     fontsize=14,
                     arrowprops=dict(facecolor='black', shrink=0.1)
                    )
        plt.legend()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf) 


    def hierarchical_analysis(self, n_clusters=10):

        # get data from s3
        X_train_ae = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'Mnist_ae.csv')
        X_train_ae_tsne = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'Mnist_ae_tsne.csv')
      
        agglo = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean', linkage = 'complete')
        agglo.fit_predict(X_train_ae)
        y_agglo = agglo.fit_predict(X_train_ae)

        # visualize cluster
        unu.compare_truelabel_cluster(X_train_ae_tsne.values, self.y_train, y_agglo)





