# Python ≥3.5 is required
import sys
import os
from io import StringIO
from pathlib import Path
import joblib
import boto3
import pickle
from bs4 import BeautifulSoup
import unicodedata
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
import datetime

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network


# Scikit-Learn ≥0.20 is required
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Dataframe manipulation
import numpy as np
import pandas as pd

import spacy
#spacy.download('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()
import string
import re
import langdetect

## for word embedding
import gensim
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import wordcloud

# deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.patheffects as PathEffects

#  defined utility functions
from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf


lst_stopwords = nltk.corpus.stopwords.words("english")
hotel_rec_scaler  = os.path.join(cf.ANALYSIS_PATH, 'hotel_rec_scaler.pkl')
hotel_stacked_ae_file = os.path.join(cf.TRAINED_MODEL_PATH, 'hotel_stacked_ae.pkl')


class HotelRecommendation:

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
    def load_dataset(self, city):
        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "airbnb_train_corpus.csv")


    ##############################################################################################
    # create_corpus
    ##############################################################################################
    def create_corpus(self, city):

        # read review file
        st.markdown('#### 5 samples of reviews data')
        file_name="/".join([cf.S3_DATA_BOOKING, city, 'review.csv'])
        review_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH, file_name=file_name, type='s3')
        st.write(review_df[['accommodation_id', 'review']].head())
 
        # detect language
        st.markdown('#### Detect language and select English reviews')
        review_df["text_clean"] = review_df["review"].apply(lambda x: utils_preprocess_text(x, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords))
        review_df['lang'] = review_df["text_clean"].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")
        review_df = review_df[review_df["lang"]=="en"]
        st.write(review_df[['accommodation_id', 'review', 'text_clean', 'lang']].head())

        # Split review to sentences & clauses and preprocessing 
        st.markdown('#### Creating corpus - Split review to sentences and clauses')
        sentences = []
        clause_clean = []
        listing_ids = []
        clause_list = []
        clause_listing_ids = []
        word_count = []

        st.write("Number of reviews: ", str(len(review_df)))
        for i in range(0, len(review_df.review)):
            if (i % 500 == 0):
                st.write('processing review number: ' + str(i))
            doc = nlp(review_df.review.iloc[i])
            listing_id = review_df.accommodation_id.iloc[i]
            for sent in doc.sents:
                #lst_tokens = nltk.tokenize.word_tokenize(text_clean)
                #if(len(lst_tokens) > 4):
                sentences.append(sent.text)
                listing_ids.append(listing_id) 
                sent = str(sent).lower().strip()
                clause_pos, words_sentence = self.create_clause(sent)
                if len(clause_pos) > 1:
                    for j in range(0, len(clause_pos)-1):
                        start = clause_pos[j]
                        end = clause_pos[j+1] - 1
                        clause = ' '.join(words_sentence[start:end+1])
                        text_clean = utils_preprocess_text(clause, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
                        clause_listing_ids.append(listing_id)
                        clause_list.append(clause)
                        clause_clean.append(text_clean)
                        word_count.append(end - start)

        # create data frame to store corpus
        corpus_df = pd.DataFrame()
        corpus_df['listing_id'] = listing_ids
        corpus_df['text'] = sentences

        clause_df = pd.DataFrame()
        clause_df['listing_id'] = clause_listing_ids
        clause_df['text_clean'] = clause_clean
        clause_df['text'] = clause_list
        clause_df['word_count'] = word_count
        # select clauses with number of words > 7 => to reduce noise and unclear info
        clause_df = clause_df[clause_df['word_count'] > 7]
        clause_df = clause_df.reset_index(drop=True)
        st.markdown('#### Finish splitting reviews to clauses')
        st.write("Number of clauses: ", clause_df.shape)
        st.write("Examples of first 10 clauses")
        st.write(clause_df.head(10))

        # store corpus to csv file
        st.markdown('#### Storing corpus to S3')
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, 'corpus_df.csv']), 
                          data=corpus_df, type='s3')
        # store corpus to csv file
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, 'clause_df.csv']), 
                          data=clause_df, type='s3')
        st.markdown('#### Finished storing corpus')


    def create_clause(self, sent):
        doc = nlp(sent)
        clause_pos = [0]
        words_sentence = []
        i = 0
        for token in doc: 
            #print(i, ": ", token.text, "-->",token.dep_,"-->", token.pos_)
            words_sentence.append(token.text)
            if(token.dep_.startswith("nsubj") == True):
                if(i > 0):
                    clause_pos.append(i)
            if((token.dep_ == "punct") & (token.pos_ == "NOUN")):
                clause_pos.append(i)
            i = i + 1
        clause_pos.append(i)    
        return clause_pos, words_sentence    



    ##############################################################################################
    # create_word_embedding
    ##############################################################################################
    def create_word_embedding(self, city, corpus_df):

        # Set values for various parameters
        feature_size = 300    # Word vector dimensionality  
        window_context = 30          # Context window size                                                                                    
        min_word_count = 1   # Minimum word count                        
        sample = 1e-3   # Downsample setting for frequent words

        st.markdown('#### Transforming text to numeric features using word embedding technique')

        # creating embedding data
        tokenized_corpus = []
        corpus_df['text_clean'] = corpus_df['text_clean'].astype(str)
        for words in corpus_df['text_clean']:
            words = str(words)
            tokenized_corpus.append(words.split())

        # download and train pretrained word embedding
        EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
        cf.S3_CLIENT.download_file(cf.S3_DATA_PATH, "/".join([cf.S3_DATA_BOOKING, EMBEDDING_FILE]), EMBEDDING_FILE)
        pretrained_model = Word2Vec(size = feature_size, window=window_context, min_count = min_word_count, workers=-1)
        pretrained_model.build_vocab(tokenized_corpus)
        pretrained_model.intersect_word2vec_format(EMBEDDING_FILE, lockf=1.0, binary = True)
        pretrained_model.train(tokenized_corpus, total_examples=pretrained_model.corpus_count, epochs = 5)
        
        # cf.S3_CLIENT.download_file(cf.S3_DATA_PATH, "/".join([cf.S3_DATA_BOOKING, 'booking_pretrained_model.pkl']), 'booking_pretrained_model.pkl')
        #pretrained_model = joblib.load('booking_pretrained_model.pkl')
        #pretrained_model.train(tokenized_corpus, total_examples=pretrained_model.corpus_count, epochs = 5)       
        # store pretrained word embedding
        joblib.dump(pretrained_model, city + '_booking_pretrained_model.pkl')

        # train model
        embeddings =  self.vectorize(corpus_df['text_clean'], pretrained_model)
        embeddings_df = pd.DataFrame(embeddings)
        st.markdown('#### Review embedding word vectors')
        st.write(embeddings_df.head()) 
        st.write(embeddings_df.shape)     

        # scaling data for model training
        scaler = MinMaxScaler()
        embeddings_df = pd.DataFrame(scaler.fit_transform(embeddings_df), columns = range(0,300))
        st.markdown('#### Review scaled embedding word vectors')
        st.write(embeddings_df.head()) 
        joblib.dump(scaler, hotel_rec_scaler)

        # store embedding data to csv file
        st.markdown('#### Storing data to S3')
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, 'embeddings_df.csv']), 
                          data=embeddings_df, type='s3')
        


    ##############################################################################################
    # text_processing
    ##############################################################################################
    def text_processing(self, city):
        # create corpus
        st.markdown('<p style="color:Green; font-size: 30px;"> 1. Text Processing and Normalization</p>', unsafe_allow_html=True)
        self.create_corpus(city)
        file_name="/".join([cf.S3_DATA_BOOKING, city, 'clause_df.csv'])
        clause_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH, file_name=file_name, type='s3')
        st.write(clause_df.shape)
        st.write(clause_df.head(10))
            
        # create word embedding
        st.markdown('<p style="color:Green; font-size: 30px;"> 2. Feature Extraction - Word Embedding</p>', unsafe_allow_html=True)
        self.create_word_embedding(city, clause_df)



    ##############################################################################################
    # Dimensionality Reduction with Auto Encoder
    ##############################################################################################
    def train_autoencoder(self, city, encoding1_dim=50, encoding2_dim=600, latent_dim=15, epochs=30):

        # read word embedding data
        file_name = "/".join([cf.S3_DATA_BOOKING, city, 'embeddings_df.csv'])
        X_train = dm.read_csv_file(cf.S3_DATA_PATH, file_name, type='s3')

        # setup
        tf.random.set_seed(42)
        np.random.seed(42)
        keras.backend.clear_session()
        tf.keras.backend.clear_session()

        input_dim = X_train.shape[1]
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
        # nb_epoch = cf.data['autoencoder_n_epoch']
        autoencoder.compile(optimizer='adam', loss='mse')

        cp = ModelCheckpoint(filepath='autoencoder1.h5', save_best_only=True, verbose=0)
        tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        t_ini = datetime.datetime.now()
        history = autoencoder.fit(X_train, 
                                  X_train, 
                                  epochs=epochs, 
                                  #batch_size=batch_size, 
                                  shuffle=True, 
                                  validation_split=0.2, 
                                  verbose=1
        ).history
        t_fin = datetime.datetime.now()

        st.markdown('#### Training history')
        st.write('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))
        df_history = pd.DataFrame(history) 
        # plot training history
        st.write("")    
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(df_history['loss'], linewidth=2, label='Train')
        plt.plot(df_history['val_loss'], linewidth=2, label='Test')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # predict embedded encoder layer
        encoded = encoder_layer.predict(X_train)
        ae_embeddings = pd.DataFrame(encoded, columns = range(0,latent_dim))
        st.markdown('#### Review encoder embedding features')
        st.write(ae_embeddings.shape)
        st.write(ae_embeddings.head())

        # save model
        keras.models.save_model(autoencoder, cf.TRAINED_MODEL_PATH + '/' + city + "_booking_autoencoder_model.h5")

        return ae_embeddings


    ##############################################################################################
    # Topic Modeling
    ##############################################################################################

    def kmeans_topic_modeling(self, city, ae_embeddings, n_clusters=8):

        st.markdown('#### Training clustering using KMeans')
        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        kmeans.fit(ae_embeddings)
        cluster_labels = kmeans.labels_
        cluster_df = pd.DataFrame()
        cluster_df['Cluster'] = cluster_labels
        ae_embeddings['Cluster'] = cluster_labels

        st.markdown('#### Applying TSNE to transform embedding features to 2 dimensions and plot data with cluster')
        tsne = TSNE(n_components=2, random_state=9)
        if(len(ae_embeddings) > 30000):
            tsne_sample = ae_embeddings.sample(n=30000, random_state=1)
        else:
            tsne_sample = ae_embeddings
        tsne_sample = tsne_sample.reset_index(drop=True)
        X_train_tsne = tsne.fit_transform(tsne_sample.drop(['Cluster'], axis=1))
        X_train_tsne = pd.DataFrame(X_train_tsne)
        X_train_tsne['Cluster'] =  tsne_sample['Cluster']
        plot_cluster(X_train_tsne[[0,1]].values, X_train_tsne['Cluster'], n_clusters)

        st.markdown('#### Review 20 clauses in each cluster')
        file_name = "/".join([cf.S3_DATA_BOOKING, city, 'clause_df.csv'])
        clause_df = dm.read_csv_file(cf.S3_DATA_PATH, file_name, type='s3')
        clause_df = clause_df.reset_index(drop=True)
        clause_df['Cluster'] = cluster_df['Cluster']
        self.show_cluster(clause_df, n_clusters)
        
        st.markdown('#### Storing auto encoder embedding data and cluster data to S3')
        # store embedding data to csv file
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, 'booking_cluster_df.csv']), 
                          data=clause_df, type='s3')
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, 'booking_ae_embedding_df.csv']), 
                          data=ae_embeddings, type='s3')
        


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


    def show_cluster(self, corpus_cluster_df, n_clusters):
        n_records = 10
        for i in range(0, n_clusters):          
            st.write("**Cluster number**: ", str(i))
            for j in range(0, n_records):
                st.write(corpus_cluster_df[(corpus_cluster_df.Cluster == i)].text.iloc[j])
                st.write()
            self.plot_wordcloud(corpus=corpus_cluster_df[(corpus_cluster_df.Cluster == i)].text_clean,
                                max_words=500, max_font_size=35, figsize=(10,5))


    def plot_wordcloud(self, corpus, max_words=150, max_font_size=35, figsize=(10,10)):
        wc = wordcloud.WordCloud(background_color='black', max_words=max_words, max_font_size=max_font_size)
        wc = wc.generate(str(corpus)) #if type(corpus) is not dict else wc.generate_from_frequencies(corpus)     
        fig, ax = plt.subplots(figsize=figsize)
        plt.axis('off')
        plt.imshow(wc, cmap=None)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    ##############################################################################################
    # Recommendaton
    ##############################################################################################
    def vectorize(self,corpus, model):

        # global embeddings
        embeddings = []
        #a list to store the vectors; these are vectorized Netflix Descriptions
        for line in corpus: #for each cleaned description
            w2v = None
            count = 0
            for word in line.split():
                if word in model.wv.vocab:
                    count += 1
                    if w2v is None:
                        w2v = model.wv[word]
                    else:
                        w2v = w2v + model.wv[word]
            if w2v is not None:
                w2v = w2v / count
                #append element to the end of the embeddings list 
                embeddings.append(w2v)

        return embeddings 


    def calculate_similarity(self, item, pretrained_model, embeddings, cluster_df, encoder_layer):

        vectorized_item = [item]
        vectorized_item = self.vectorize(vectorized_item, pretrained_model)
        embedded_value = pd.DataFrame(vectorized_item, columns = range(0,300))
        scaler = joblib.load(hotel_rec_scaler)
        embedded_value = scaler.transform(embedded_value)
        embedded_value = encoder_layer.predict(embedded_value)
        cosine_sim = cosine_similarity(embeddings,embedded_value)
        cluster_df['Cosine'] = cosine_sim
        cluster_df = cluster_df.sort_values(['Cosine'], ascending=False)
        cluster_df = cluster_df.reset_index(drop=True)        

        return cluster_df


    def max_similarity(self, similarity_df, topic):
        sim_df = similarity_df.groupby(['listing_id']).Cosine.max()
        sim_df = pd.DataFrame(sim_df).reset_index(drop=False)
        sim_df.columns = ['listing_id', 'score' + str(topic)]
        # sim_df['count'+ str(topic)] = similarity_df.groupby(['listing_id']).listing_id.count().values
        #sim_df = sim_df.sort_values('score', ascending=False)
        return sim_df


    def hotel_recommendation_booking(self, city, 
                                    cluster0='', 
                                    cluster1='', 
                                    cluster2='', 
                                    cluster3='', 
                                    cluster4='', 
                                    cluster5=''):
        # text processing and normalizing input data
        cluster0 = utils_preprocess_text(cluster0, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
        cluster1 = utils_preprocess_text(cluster1, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
        cluster2 = utils_preprocess_text(cluster2, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
        cluster3 = utils_preprocess_text(cluster3, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
        cluster4 = utils_preprocess_text(cluster4, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)
        cluster5 = utils_preprocess_text(cluster5, flg_stemm=True, flg_lemm=True, lst_stopwords=lst_stopwords)

        #  load corpus
        train_vars = range(0,15)
        train_vars = [str(var) for var in train_vars]
        corpus_file="/".join([cf.S3_DATA_BOOKING, city, 'clause_df.csv'])
        corpus_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH,  file_name=corpus_file, type='s3')

        # load pretrain model
        pretrained_model = joblib.load(city + '_booking_pretrained_model.pkl')
        
        # load auto encoder embedding data
        embedding_file = "/".join([cf.S3_DATA_BOOKING, city, 'booking_ae_embedding_df.csv'])
        ae_embeddings = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH,  file_name=embedding_file, type='s3')

        # load encoder embedding model  
        autoencoder = keras.models.load_model(cf.TRAINED_MODEL_PATH + '/' + city + "_booking_autoencoder_model.h5")
        encoder_layer = Model(autoencoder.input, autoencoder.layers[-4].output)

        # calculate similarity of each condition
        st.markdown('<p style="color:Green; font-size: 30px;"> Best match for each condition: </p>', unsafe_allow_html=True)
        df0 = self.calculate_similarity(cluster0, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        st.write("Best match condion 1: listing_id ", df0.listing_id.iloc[0], ' - ', df0.text.iloc[0], " - cosine = ", df0.Cosine.iloc[0])
        df1 = self.calculate_similarity(cluster1, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        st.write("Best match condion 2: listing_id ", df1.listing_id.iloc[0], ' - ', df1.text.iloc[0], " - cosine = ", df1.Cosine.iloc[0])
        df2 = self.calculate_similarity(cluster2, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        st.write("Best match condion 3: listing_id ", df2.listing_id.iloc[0], ' - ', df2.text.iloc[0], " - cosine = ", df2.Cosine.iloc[0])
        df3 = self.calculate_similarity(cluster3, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        st.write("Best match condion 4: listing_id ", df3.listing_id.iloc[0], ' - ', df3.text.iloc[0], " - cosine = ", df3.Cosine.iloc[0])
        df4 = self.calculate_similarity(cluster4, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        st.write("Best match condion 5: listing_id ", df4.listing_id.iloc[0], ' - ', df4.text.iloc[0], " - cosine = ", df4.Cosine.iloc[0])
        df5 = self.calculate_similarity(cluster5, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        st.write("Best match condion 6: listing_id ", df5.listing_id.iloc[0], ' - ', df5.text.iloc[0], " - cosine = ", df5.Cosine.iloc[0])


        sim0_df = self.max_similarity(df0, 0)
        sim1_df = self.max_similarity(df1, 1)
        sim2_df = self.max_similarity(df2, 2)
        sim3_df = self.max_similarity(df3, 3)
        sim4_df = self.max_similarity(df4, 4)
        sim5_df = self.max_similarity(df5, 5)

        sum_sim_df = pd.DataFrame(corpus_df['listing_id'].unique(), columns =['listing_id'])
        sum_sim_df = sum_sim_df.merge(sim0_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim1_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim2_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim3_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim4_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim5_df, how='left', left_on='listing_id', right_on='listing_id')


        sum_sim_df.fillna(0, inplace=True)
        sum_sim_df['avg_sim'] = sum_sim_df[['score0', 'score1', 'score2', 'score3', 'score4','score5']].mean(axis=1)
        sum_sim_df = sum_sim_df.sort_values(['avg_sim'], ascending=False)
        st.markdown('<p style="color:Green; font-size: 30px;"> Top 10 matched hotels: </p>', unsafe_allow_html=True)
        st.write(sum_sim_df.head(10))

        st.markdown('<p style="color:Green; font-size: 30px;"> Review best matched hotel: </p>', unsafe_allow_html=True)
        max_sim_id = sum_sim_df.head(1)['listing_id'].iloc[0]
        st.write('Listing_id: ',max_sim_id)
        st.write(df0[df0.listing_id == max_sim_id].text.iloc[0])
        st.write(df1[df1.listing_id == max_sim_id].text.iloc[0])
        st.write(df2[df2.listing_id == max_sim_id].text.iloc[0])
        st.write(df3[df3.listing_id == max_sim_id].text.iloc[0])
        st.write(df4[df4.listing_id == max_sim_id].text.iloc[0])
        st.write(df5[df5.listing_id == max_sim_id].text.iloc[0])



    ##############################################################################################
    # Knowledge Graph
    ##############################################################################################
    def extract_pattern1(self, doc):
        
        adj = ""
        obj1 = []
        obj2 = []
        for i, tok in enumerate(doc):
            #print(i, ": ", tok.text, "-->",tok.dep_,"-->", tok.pos_)
            
            # structure adj + Noun: capture adj or gerund or Participant as adj ()
            # if((tok.dep_.endswith("acomp") == False) & (tok.pos_.endswith("ADJ") == True)):
            if(tok.pos_.endswith("ADJ") == True):
                adj = tok.text
            elif((tok.dep_.endswith("amod") == True) & (tok.pos_.endswith("VERB") == True)):
                adj = tok.text
                
                
            if((tok.pos_.endswith("NOUN")==True) | (tok.pos_.endswith("PROPN") == True)| (tok.pos_.endswith("PRON") ==True)):
                if(tok.dep_.endswith("compound") == False):
                    entity = tok.text
                    if(len(adj) > 0):
                        obj1.append(entity)
                        obj2.append(adj)
                    adj = ""
        
        return obj1, obj2


    def extract_pattern2(self, doc):
        
        entity = ""
        obj1 = []
        obj2 = []
        flag = 0
        for i, tok in enumerate(doc):
            # print(i, ": ", tok.text, "-->",tok.dep_,"-->", tok.pos_)
                
            # extract subject/ root
            if((tok.pos_.endswith("NOUN")==True) | (tok.pos_.endswith("PROPN") == True)| (tok.pos_.endswith("PRON") ==True)):
                if((tok.dep_.endswith("ROOT") == True) | (tok.dep_.endswith("appos") == True) | (tok.dep_.find("subj") == True)):
                    entity = tok.text
                else:
                    entity = ""
                    
            if((tok.pos_.endswith("AUX")==True)):
                flag = 1
                    
            # structure adj + Noun: capture adj or gerund or Participant as adj ()
            # if((tok.dep_.endswith("acomp") == False) & (tok.pos_.endswith("ADJ") == True)):
            if(tok.pos_.endswith("ADJ") == True):
                adj = tok.text
                if((len(entity) > 0) & (flag==1)):
                    obj1.append(entity)
                    obj2.append(adj)
                entity = ""
                
        
        return obj1, obj2

    def extract_pattern3(self, doc):
        
        entity1 = ''
        entity2 = ''
        obj1 = []
        obj2 = []
        flag = 0
        for i, tok in enumerate(doc):
            # print(i, ": ", tok.text, "-->",tok.dep_,"-->", tok.pos_)
                
            # extract subject/ root
            if((tok.pos_.endswith("NOUN")==True) | (tok.pos_.endswith("PROPN") == True)| (tok.pos_.endswith("PRON") ==True)):
                if((tok.dep_.endswith("ROOT") == True) | (tok.dep_.endswith("appos") == True) | (tok.dep_.find("subj") == True)):
                    entity1 = tok.text
                else:
                    entity2 = tok.text
            
            if((len(entity1) > 0) & (len(entity2) > 0)):
                #print(entity1, entity2)
                obj1.append(entity1)
                obj2.append(entity2)
                entity2= ''
        
        return obj1, obj2


    def extract_info(self, clause):
        house_obj1 = []
        house_obj2 = []
        obj1, obj2 = self.extract_pattern1(clause)
        house_obj1 = house_obj1 + obj1
        house_obj2 = house_obj2 + obj2
        obj1, obj2 = self.extract_pattern2(clause)
        house_obj1 = house_obj1 + obj1
        house_obj2 = house_obj2 + obj2
        obj1, obj2 = self.extract_pattern3(clause)
        house_obj1 = house_obj1 + obj1
        house_obj2 = house_obj2 + obj2

    
        return house_obj1, house_obj2


    
    def generate_graph_data(self, city, listing_id, cluster_id):
        file_name = "/".join([cf.S3_DATA_BOOKING, city, 'booking_cluster_df.csv'])
        house_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH, file_name=file_name, type='s3')
        house_df = house_df[house_df.listing_id == listing_id].reset_index(drop=True)
        house_df = house_df[house_df.Cluster == cluster_id].reset_index(drop=True)
        house_obj1 = []
        house_obj2 = []
        graph_df = pd.DataFrame()

        for i in range(0, len(house_df)):
            sent = house_df.text[i]
            #sent = utils_preprocess_text(sent)
            obj1, obj2 = self.extract_info(nlp(sent))
            house_obj1 = house_obj1 + obj1
            house_obj2 = house_obj2 + obj2

        graph_df['obj1'] = house_obj1
        graph_df['obj2'] = house_obj2
        graph_df['ID'] = range(0,len(graph_df))
        graph_df = graph_df[graph_df.obj2 != 'NA']

        graph_df['obj1_type'] = graph_df['obj1'].apply(lambda x:nlp(x)[0].pos_)
        graph_df['obj2_type'] = graph_df['obj2'].apply(lambda x:nlp(x)[0].pos_)

        lem = nltk.stem.wordnet.WordNetLemmatizer()
        graph_df['obj1'] = graph_df['obj1'].apply(lambda x: str.lower(x))
        graph_df['obj2'] = graph_df['obj2'].apply(lambda x: str.lower(x))
        graph_df['obj1'] = graph_df['obj1'].apply(lambda x: lem.lemmatize(x))
        graph_df['obj2'] = graph_df['obj2'].apply(lambda x: lem.lemmatize(x))
            
        return house_df,  graph_df

    
    def extract_entities(self, graph_df):
        entity_obj1 = list(graph_df[(graph_df.obj1_type == 'PROPN') | (graph_df.obj1_type == 'NOUN')].obj1.values)
        entity_obj2 = list(graph_df[(graph_df.obj2_type == 'PROPN') | (graph_df.obj2_type == 'NOUN')].obj2.values)

        entity_list = entity_obj1 + entity_obj2
        entity_list = pd.DataFrame(entity_list, columns = ['entity']).value_counts().reset_index()
        entity_list.rename(columns = {0: 'count'}, inplace=True)

        return list(entity_list.entity.values)


    # Define function to generate Pyvis visualization
    
    def generate_network_viz(self, df, source_col, target_col, weights, 
                             layout='barnes_hut',
                             central_gravity=0.15,
                             node_distance=420,
                             spring_length=100,
                             spring_strength=0.15,
                             damping=0.96
                             ):
       
        # Create networkx graph object from pandas dataframe
        G = nx.from_pandas_edgelist(df, source_col, target_col, weights)

        # Initiate PyVis network object
        drug_net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white')
        # drug_net = generate_network_viz(df_db_int_sm, 'drug_1_name', 'drug_2_name', 'weight', layout='repulsion')

        # Take Networkx graph and translate it to a PyVis graph format
        drug_net.from_nx(G)

        # Generate network with specific layout settings
        drug_net.repulsion(node_distance=420, central_gravity=0.33,
                           spring_length=110, spring_strength=0.10,
                           damping=0.95)


        return drug_net


    def generate_graph_booking(self, city, listing_id, cluster_id, node_degree=0):
        
        house_df,  graph_df = self.generate_graph_data(city, listing_id, cluster_id)
        weight_graph_df = pd.DataFrame(graph_df.groupby(['obj1','obj2']).count()).reset_index()
        weight_graph_df.rename(columns={'ID':'weight'}, inplace=True)
        weight_graph_df = weight_graph_df.drop_duplicates(subset=['obj1','obj2'], keep='last')

        # Generate a networkx graph based on subset data
        net_repulsion = self.generate_network_viz(weight_graph_df, 'obj1','obj2', 'weight', layout='repulsion')

        G = nx.from_pandas_edgelist(weight_graph_df, 'obj1', 'obj2')
        selected_nodes = []
        for e in dict(G.degree()).items():
            node, degree = e
            if degree > node_degree:
                # st.write(node, ": ", degree)
                selected_nodes.append(node)
        weight_graph_df = weight_graph_df.loc[weight_graph_df['obj1'].isin(selected_nodes) | weight_graph_df['obj2'].isin(selected_nodes)]
        net_repulsion = self.generate_network_viz(weight_graph_df, 'obj1','obj2', 'weight', layout='repulsion')

        node_color = {'NOUN':'lightblue', 'PRON':'lightblue', 'PROPN':'yellow', 'VERB': 'red', 'ADJ':'red', 
                      'ADV':'red', 'DET':'red', 'X':'grey', 'INTJ':'grey','AUX':'grey','NUM':'grey',
                      'ADP':'grey','PUNCT':'grey', 'SCONJ':'grey'}

        node_list = []
        for i in range(0, len(net_repulsion.nodes)):
            node_list.append(net_repulsion.nodes[i]['id'])
        type_node = [nlp(x)[0].pos_ for x in node_list ]

        for i in range(0, len(net_repulsion.nodes)):
            net_repulsion.nodes[i]['color'] = node_color[type_node[i]]


        # Save and read graph as HTML file (on Streamlit Sharing)
        html_file = os.path.join(cf.IMAGE_PATH, "pyvis_graph.html")
        try:
            net_repulsion.save_graph(html_file)
            HtmlFile = open(html_file, 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            net_repulsion.save_graph(html_file)
            HtmlFile = open(html_file, 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=1000)

        st.write(house_df.shape)
        for i in range(0, len(house_df)):
            st.write(house_df.text[i])



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


def plot_cluster(X, y, n_cluster):

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_cluster))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,7))
    ax[0].scatter(X[:, 0], X[:, 1], s=3)
    sc = ax[1].scatter(X[:,0], X[:,1], s=3,c=y, cmap="jet")
    #ax.axis('off')
    #ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_cluster):
        # Position of each label.
        xtext, ytext = np.median(X[y == i, :], axis=0)
        txt = ax[1].text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
                PathEffects.Stroke(linewidth=3, foreground="w"),
                PathEffects.Normal()])
        txts.append(txt)
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

