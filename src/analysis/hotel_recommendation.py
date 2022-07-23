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

import scipy

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


    def calculate_similarity(self, search_condition, model, cluster_df):

        dic_bow = model.fit_transform(cluster_df['text'])
        bow_df = dic_bow.todense()
        bow_search_string = model.transform(search_condition)
        bow_search_string = bow_search_string.todense()
        bow_search_string = bow_search_string * 2
        bow_search_string = bow_search_string - 1
        similarity_list = []
        for i in range(0, len(cluster_df)):
            similarity = 1  -  scipy.spatial.distance.cdist(np.array(bow_df[i]), np.array(bow_search_string), 'hamming')
            similarity_list.append(similarity.sum())       

        return similarity_list


    def max_similarity(self, similarity_df, topic):
        sim_df = similarity_df.groupby(['listing_id']).Cosine.max()
        sim_df = pd.DataFrame(sim_df).reset_index(drop=False)
        sim_df.columns = ['listing_id', 'score' + str(topic)]
        # sim_df['count'+ str(topic)] = similarity_df.groupby(['listing_id']).listing_id.count().values
        #sim_df = sim_df.sort_values('score', ascending=False)
        return sim_df


    def search_corpus(self, city, search_string):
        #  load corpus
        corpus_file="/".join([cf.S3_DATA_BOOKING, city, 'corpus_df.csv'])
        corpus_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH,  file_name=corpus_file, type='s3')
        corpus_df['text'] = corpus_df['text'].astype(str)
        corpus_df['text'] = corpus_df['text'].apply(lambda x: str(x).lower())

        search_corpus = []
        cluster_df = pd.DataFrame()

        search_string = search_string.lower()
        search_string = re.sub(r'[^\w\s]', '', str(search_string).lower().strip())
        search_corpus = nltk.tokenize.word_tokenize(search_string)  
        st.write(search_corpus) 
        for e in search_corpus:
            voc_df = corpus_df[corpus_df['text'].str.contains(e)]
            st.write(voc_df.shape)
            cluster_df = cluster_df.append(voc_df)
            cluster_df = cluster_df.drop_duplicates()

        return search_corpus,  cluster_df





    def hotel_recommendation_booking(self, city, 
                                    search_string1='', 
                                    search_string2='', 
                                    search_string3='', 
                                    search_string4='', 
                                    search_string5=''):
        # text processing and normalizing input data
        search_string1, cluster_df1 = self.search_corpus(city, search_string1)
        search_string2, cluster_df2 = self.search_corpus(city, search_string2)
        search_string3, cluster_df3 = self.search_corpus(city, search_string3)
        search_string4, cluster_df4 = self.search_corpus(city, search_string4)
        search_string5, cluster_df5 = self.search_corpus(city, search_string5)

        st.write(len(cluster_df1), len(cluster_df2), len(cluster_df3), len(cluster_df4), len(cluster_df5))

        #  load corpus
        corpus_file="/".join([cf.S3_DATA_BOOKING, city, 'corpus_df.csv'])
        corpus_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH,  file_name=corpus_file, type='s3')
        corpus_df['text'] = corpus_df['text'].astype(str)
        corpus_df['text'] = corpus_df['text'].apply(lambda x: str(x).lower())

        vectorizer = CountVectorizer(max_features=10000, ngram_range=(1,1), lowercase=False, binary='true')


        # calculate similarity of each condition
        st.markdown('<p style="color:Green; font-size: 30px;"> Best match for each condition: </p>', unsafe_allow_html=True)
        cluster_df1['score1'] = self.calculate_similarity(search_string1, vectorizer, cluster_df1)
        cluster_df1 = cluster_df1.sort_values(['score1'], ascending=False).reset_index(drop=True)
        sim_df1 = cluster_df1.groupby(['listing_id']).score1.max()
        sim_df1 = pd.DataFrame(sim_df1).reset_index(drop=False)
        for j in range(0, 3):
            st.write('condition 1')
            st.write(j, cluster_df1.text[j], cluster_df1.score1[j])

        cluster_df2['score2'] = self.calculate_similarity(search_string2, vectorizer, cluster_df2)
        cluster_df2 = cluster_df2.sort_values(['score2'], ascending=False).reset_index(drop=True)
        sim_df2 = cluster_df2.groupby(['listing_id']).score2.max()
        sim_df2 = pd.DataFrame(sim_df2).reset_index(drop=False)
        for j in range(0, 3):
            st.write('condition 2')
            st.write(j, cluster_df2.text[j], cluster_df2.score2[j])

        cluster_df3['score3'] = self.calculate_similarity(search_string3, vectorizer, cluster_df3)
        cluster_df3 = cluster_df3.sort_values(['score3'], ascending=False).reset_index(drop=True)
        sim_df3 = cluster_df3.groupby(['listing_id']).score3.max()
        sim_df3 = pd.DataFrame(sim_df3).reset_index(drop=False)
        for j in range(0, 3):
            st.write('condition 3')
            st.write(j, cluster_df3.text[j], cluster_df3.score3[j])


        cluster_df4['score4'] = self.calculate_similarity(search_string4, vectorizer, cluster_df4)
        cluster_df4 = cluster_df4.sort_values(['score4'], ascending=False).reset_index(drop=True)
        sim_df4 = cluster_df4.groupby(['listing_id']).score4.max()
        sim_df4 = pd.DataFrame(sim_df4).reset_index(drop=False) 
        for j in range(0, 3): 
            st.write('condition 4')
            st.write(j, cluster_df4.text[j], cluster_df4.score4[j]) 

        cluster_df5['score5'] = self.calculate_similarity(search_string5, vectorizer, cluster_df5)
        cluster_df5 = cluster_df5.sort_values(['score5'], ascending=False).reset_index(drop=True)
        sim_df5 = cluster_df5.groupby(['listing_id']).score5.max()
        sim_df5 = pd.DataFrame(sim_df5).reset_index(drop=False)
        for j in range(0, 3): 
            st.write('condition 5')
            st.write(j, cluster_df5.text[j], cluster_df5.score5[j])


        result_df = pd.DataFrame(corpus_df['listing_id'].unique(), columns =['listing_id'])
        result_df = result_df.merge(sim_df1[['listing_id', 'score1']],how='left', on='listing_id')
        result_df = result_df.merge(sim_df2[['listing_id', 'score2']],how='left', on='listing_id')
        result_df = result_df.merge(sim_df3[['listing_id', 'score3']],how='left', on='listing_id')
        result_df = result_df.merge(sim_df4[['listing_id', 'score4']],how='left', on='listing_id')
        result_df = result_df.merge(sim_df5[['listing_id', 'score5']],how='left', on='listing_id')
        result_df = result_df.fillna(0)
        result_df['score3'] = result_df['score3'] * 8
        result_df['avg_score'] = result_df[['score1','score2','score3','score4','score5']].mean(axis=1)
        result_df = result_df.sort_values('avg_score', ascending=False)

        st.write('The hotel best match: ')
        listing_id = result_df.listing_id.iloc[0]
        st.write(cluster_df1.loc[cluster_df1.listing_id == listing_id].text.iloc[0])
        st.write(cluster_df2.loc[cluster_df2.listing_id == listing_id].text.iloc[0])
        st.write(cluster_df3.loc[cluster_df3.listing_id == listing_id].text.iloc[0])
        st.write(cluster_df4.loc[cluster_df4.listing_id == listing_id].text.iloc[0])
        st.write(cluster_df5.loc[cluster_df5.listing_id == listing_id].text.iloc[0])


        return result_df



    ##############################################################################################
    # Knowledge Graph
    ##############################################################################################


    def extract_pattern(self, doc):
        
        obj_list = []
        adj_list = []

        
        flag = 0
        for i, tok in enumerate(doc):
            
            
            # nearby words on the left
            if((tok.pos_.endswith("NOUN")==True) | (tok.pos_.endswith("PROPN") == True)):
                obj_list.append(tok.text)
            else:
                obj_list.append('N/A')
                
            if(tok.pos_.endswith("ADJ")==True):
                adj_list.append(tok.text)
            else:
                adj_list.append('N/A')
                
        return obj_list, adj_list



    def extract_info(self, clause):
        list_obj1 = []
        list_obj2 = []

        for e in range(0, len(cluster_df)):
            doc = nlp(cluster_df.text.iloc[e])
            obj_list, adj_list = self.extract_pattern(doc)
            
            for i in range(0, len(obj_list)):
                if (obj_list[i] != 'N/A'):
                    #print(i, obj_list[i])
                    # words on the left
                    #print('left')
                    for j in range(i-5, i-1):
                        if((j < i) & (j > i - 5) & (j > 0 )):
                            if (obj_list[j] != 'N/A'):
                                #print(i, j)
                                #print(obj_list[i], obj_list[j])
                                list_obj1.append(obj_list[i])
                                list_obj2.append(obj_list[j])
                    # words on the right    
                    #print('right')
                    for j in range(i+1, i+5):
                        if((j > i) & (j < i + 5) & (j < len(obj_list))):
                            if (obj_list[j] != 'N/A'):
                                #print(i, j)
                                #print(obj_list[i], obj_list[j])
                                list_obj1.append(obj_list[i])
                                list_obj2.append(obj_list[j])
            
                    for j in range(i-3, i):
                        if((j < i) & (j > i - 3) & (j >= 0 )):
                            if (adj_list[j] != 'N/A'):
                                #print(i, j)
                                #print(obj_list[i], adj_list[j])
                                list_obj1.append(obj_list[i])
                                list_obj2.append(adj_list[j])
                    # words on the right    
                    #print('right')
                    for j in range(i+1, i+3):
                        if((j > i) & (j < i + 3) & (j < len(obj_list))):
                            if (adj_list[j] != 'N/A'):
                                #print(i, j)
                                #print(obj_list[i], adj_list[j])
                                list_obj1.append(obj_list[i])
                                list_obj2.append(adj_list[j])


    
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

