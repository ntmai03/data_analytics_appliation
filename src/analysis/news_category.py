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
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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


news_scaler  = os.path.join(cf.ANALYSIS_PATH, 'news_scaler.pkl')


    

class NewsCategory:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    TARGET_VALUE = {'TRAVEL':0, 'FOOD & DRINK':1, 'BUSINESS':2, 'SPORTS':3}
    RAW_VARS = ['category', 'headline', 'short_description', 'text']
    TARGET = 'category'
    TEXT_VAR = 'text_clean'

    def __init__(self):
 
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.load_ds()


    def load_ds(self):
        df = dm.read_csv_file(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "News_Category.csv")
        # df = df.sample(frac=1)
        df_train, df_test = model_selection.train_test_split(df, test_size=0.1, random_state=9)
        self.df_train = df_train.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)
        self.y_train  = df_train[self.TARGET]
        self.y_test  = df_test[self.TARGET]
        self.data = df


    def load_preprocessed_ds(self, data_file):
        df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + data_file)
        return df


    def load_tsne_data(self):
        df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + "News_Category_tsne_data.csv")
        self.tsne_data = df
        st.write(df.head())

    def load_ae_embedding(self):
        df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + "News_Category_ae_embbeding.csv")
        self.ae_embbeding = df
        st.write(df.head())


    def word_tokenize(self, corpus):
        wpt = nltk.WordPunctTokenizer()
        #train_corpus = [wpt.tokenize(document) for document in df_train['text_clean']]
        #test_corpus = [wpt.tokenize(document) for document in df_test['text_clean']]
        tokenized_corpus = []
        for words in corpus:
            tokenized_corpus.append(words.split())        
        
        # download and train pretrained word embedding
        EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
        #cf.S3_CLIENT.download_file(cf.S3_DATA_PATH, "/".join([cf.S3_DATA_BOOKING, EMBEDDING_FILE]), EMBEDDING_FILE)
        # Set values for various parameters
        feature_size = 300    # Word vector dimensionality  
        window_context = 30          # Context window size                                                                                    
        min_word_count = 1   # Minimum word count                        
        sample = 1e-3   # Downsample setting for frequent words
        pretrained_model = Word2Vec(size = feature_size, window=window_context, min_count = min_word_count, workers=-1)
        pretrained_model.build_vocab(tokenized_corpus)
        pretrained_model.intersect_word2vec_format(EMBEDDING_FILE, lockf=1.0, binary = True)
        pretrained_model.train(tokenized_corpus, total_examples=pretrained_model.corpus_count, epochs = 5)
        joblib.dump(pretrained_model, 'news_category_pretrained_model.pkl')



    def vectorize(self,corpus, train_flag=0):

        if(train_flag == 1):
            self.word_tokenize(corpus)

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


    def scaling_data(self, df, var_list, train_flag=0):

        data = df.copy()

        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(data[var_list])

        # persist the model for future use
        if(train_flag == 1):
            joblib.dump(scaler, news_scaler)
        scaler = joblib.load(news_scaler)

        data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)

        return data


    def transform_target(self, y):
        y = y.map(self.TARGET_VALUE)
        return y


    def preprocess_data(self, df, train_flag=0):

        # load data
        data = df.copy()

        # word tokenizer
        embeddings = pd.DataFrame(self.vectorize(data[self.TEXT_VAR], 1))
        st.write('embedding word vectorization')
        st.write(embeddings.head())

        # scaling data
        scaled_data = self.scaling_data(embeddings, embeddings.columns, train_flag)
        st.write('scaled data')
        st.write(scaled_data.head())

        # transform target value from categorical data to numeric data
        y = self.transform_target(data[self.TARGET])

        processed_data = pd.concat([scaled_data,y], axis=1)
        self.processed_data =  processed_data
  
        if(train_flag == 1):
            st.write('Storing preprocessed data in S3')
            # store embedding data to csv file
            dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                              file_name= cf.S3_DATA_PROCESSED_PATH + "News_Category_preprocessed.csv", 
                              data=processed_data, type='s3')

        return processed_data




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


    def pca_analysis(self, n_components):
        data = self.load_preprocessed_ds('News_Category_preprocessed.csv')
        self.X_train = data.drop([self.TARGET], axis=1)
        self.y_train = data[self.TARGET]
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.X_train)
        st.write('Cumulative explained variation for the first  50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

        pca_df = pd.DataFrame(pca_result)
        self.pca_data = pca_df



    def autoencoder_analysis(self, encoding1_dim=100, encoding2_dim=600, latent_dim=15):
        data = self.load_preprocessed_ds('News_Category_preprocessed.csv')
        self.X_train = data.drop([self.TARGET], axis=1)
        self.y_train = data[self.TARGET]

        # Construct architecture
        tf.random.set_seed(42)
        np.random.seed(42)
        keras.backend.clear_session()
        tf.keras.backend.clear_session()

        # No of Neurons in each layer []
        input_dim = self.X_train.shape[1]
        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(int(encoding1_dim), activation="relu")(input_layer)
        encoder = Dense(int(encoding2_dim), activation='relu')(encoder)
        encoder = Dense(int(latent_dim), activation='tanh')(encoder)
        decoder = Dense(int(encoding2_dim), activation='tanh')(encoder)
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
        history = autoencoder.fit(self.X_train, 
                                  self.X_train, 
                                  epochs=nb_epoch, 
                                  #batch_size=batch_size, 
                                  shuffle=True, 
                                  validation_split=0.2, 
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

        # save model
        keras.models.save_model(autoencoder, cf.TRAINED_MODEL_PATH + '/' + "news_category_autoencoder_model.h5")


    def autoencoder_transform(self, data,  train_flag=0):
        autoencoder = keras.models.load_model(cf.TRAINED_MODEL_PATH + '/' + "news_category_autoencoder_model.h5")
        encoder_layer = Model(autoencoder.input, autoencoder.layers[-4].output)
        ae_embeddings = encoder_layer.predict(data)
        self.ae_embeddings = pd.DataFrame(ae_embeddings)

        st.markdown("#### AutoeEncoder Transformed data")
        st.write(self.ae_embeddings.head())

        if(train_flag == 1):
            dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name=cf.S3_DATA_PROCESSED_PATH + 'News_Category_ae_embbeding.csv', data=self.ae_embeddings, type='s3')


    def kmeans_analysis(self, data, n_clusters=4, save_flag=0):

        if(save_flag == 2):
            kmeans = joblib.load(kmeans, 'news_category_kmeans.pkl')
            kmeans_cluster = kmeans.predict(data)
        elif(save_flag == 0):   
            kmeans = KMeans(n_clusters=n_clusters, random_state=9, init='k-means++')
            kmeans_cluster = kmeans.fit_predict(data)
        elif(save_flag == 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=9, init='k-means++')
            kmeans_cluster = kmeans.fit_predict(data)
            joblib.dump(kmeans, 'news_category_kmeans.pkl')
        self.kmeans = kmeans
        return kmeans_cluster


    def transform_tsne(self, data, save_flag=0):
        tsne = TSNE(n_components=2, random_state=42)
        self.tsne_data = pd.DataFrame(tsne.fit_transform(data))
        if(save_flag == 1):
            dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                              file_name=cf.S3_DATA_PROCESSED_PATH + 'News_Category_tsne_data.csv', 
                              data=self.tsne_data, type='s3')
        if(save_flag == 2):
            dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                              file_name=cf.S3_DATA_PROCESSED_PATH + 'News_Category_PCA_tsne_data.csv', 
                              data=self.tsne_data, type='s3')


    def plot_cluster(self, X, y):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
        ax[0].scatter(X[:,0], X[:,1], s=5)
        ax[1].scatter(X[:,0], X[:,1], c=y, s=5, cmap='jet')
        plt.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

    def compare_cluster(self, X, y, cluster):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
        ax[0].scatter(X[:,0], X[:,1], c=y, s=5, cmap='jet')
        ax[1].scatter(X[:,0], X[:,1], c=cluster, s=5, cmap='jet')
        plt.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)



