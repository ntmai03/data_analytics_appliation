# Python ≥3.5 is required
import sys
from pathlib import Path
import os
from io import BytesIO
import datetime

import streamlit as st
import joblib
import sys
import sklearn
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

## for data
import collections
import json

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lst_stopwords = nltk.corpus.stopwords.words("english")

import string
import re
from bs4 import BeautifulSoup
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import  LabelEncoder

# for classification
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

#from contractions import CONTRACTION_MAP
# import text_normalizer as tn

## for word embedding
import gensim
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
# deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *

from src.util import data_manager as dm
from src import config as cf


class NewsCategory:
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
        self.load_ds()


    def load_ds(self):
        df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "News_Category.csv")
        df = df.sample(frac=1)
        df.rename(columns={'category':'y'}, inplace=True)
        df_train, df_test = model_selection.train_test_split(df, test_size=0.1, random_state=9)
        self.df_train = df_train.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)
        ## get target
        self.y_train = df_train["y"]
        self.y_test = df_test["y"]

        self.data = df


    def process_data(self):
        wpt = nltk.WordPunctTokenizer()
        #train_corpus = [wpt.tokenize(document) for document in df_train['text_clean']]
        #test_corpus = [wpt.tokenize(document) for document in df_test['text_clean']]

        EMBEDDING_FILE = cf.TRAINED_MODEL_PATH + '/GoogleNews-vectors-negative300.bin.gz'
        # Set values for various parameters
        feature_size = 300    # Word vector dimensionality  
        window_context = 30          # Context window size                                                                                    
        min_word_count = 10   # Minimum word count                        
        sample = 1e-3   # Downsample setting for frequent words


        tokenized_corpus = []
        for words in self.df_train['text_clean']:
            tokenized_corpus.append(words.split())
            
        pretrained_model = Word2Vec(size = 300, window=window_context, min_count = 1, workers=-1)
        pretrained_model.build_vocab(tokenized_corpus)
        pretrained_model.intersect_word2vec_format(EMBEDDING_FILE, lockf=1.0, binary = True)
        pretrained_model.train(tokenized_corpus, total_examples=pretrained_model.corpus_count, epochs = 5)
        joblib.dump(pretrained_model, 'news_category_pretrained_model.pkl')

        embeddings =   self.vectorize(self.df_train['text_clean'])
        scaler = MinMaxScaler()
        self.X_train = pd.DataFrame(embeddings)
        self.scaled_Xtrain = pd.DataFrame(scaler.fit_transform(embeddings), columns = range(0,300))
        st.write(X_train.shape)
        st.write(X_train.head())


    def average_word_vectors(words, model, vocabulary, num_features):
        
        feature_vector = np.zeros((num_features,),dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])
        
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
            
        return feature_vector
        

    def averaged_word_vectorizer(corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        #vocabulary = list(model.wv.index_to_key)
        features = [average_word_vectors(tokenized_sentence, model.wv, vocabulary, num_features) for tokenized_sentence in corpus]
        return np.array(features)


    def vectorize(self,corpus):

        pretrained_model = joblib.load('news_category_pretrained_model.pkl')
        # global embeddings
        embeddings = []
        #a list to store the vectors; these are vectorized Netflix Descriptions
        for line in corpus: #for each cleaned description
            w2v = None
            count = 0
            for word in line.split():
                if word in pretrained_model.wv.vocab:
                    count += 1
                    if w2v is None:
                        w2v = pretrained_model.wv[word]
                    else:
                        w2v = w2v + pretrained_model.wv[word]
            if w2v is not None:
                w2v = w2v / count
                #append element to the end of the embeddings list 
                embeddings.append(w2v)

        return embeddings   


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





