# Python ≥3.5 is required
import sys
from pathlib import Path
import os
from io import StringIO
import boto3
import pickle

import streamlit as st

# Scikit-Learn ≥0.20 is required
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans

# Dataframe manipulation
import numpy as np
import pandas as pd

import joblib

import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

import spacy
#spacy.download('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()


import string
import re
from bs4 import BeautifulSoup
import unicodedata
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
import langdetect

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

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


# deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *

# Deep Learnign libraries
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.patheffects as PathEffects

from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf


'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
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
    def load_dataset(self, city):
        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "airbnb_train_corpus.csv")



    def create_corpus(self, city):
        listing_df = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/review.csv')
        st.write(listing_df[['accommodation_id', 'review']].head())
        st.write(listing_df.shape)

        lst_stopwords = nltk.corpus.stopwords.words("english")
        sentences = []
        sentences_clean = []
        listing_ids = []

        listing_df["text_clean"] = listing_df["review"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))
        # listing_df["word_count"] = listing_df['text_clean'].apply(lambda x: len(str(x).split()))
        listing_df['lang'] = listing_df["text_clean"].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")
        listing_df = listing_df[listing_df["lang"]=="en"]

        for i in range(0, len(listing_df.review)):
            doc = nlp(listing_df.review.iloc[i])
            listing_id = listing_df.accommodation_id.iloc[i]
            for sent in doc.sents:
                text_clean = utils_preprocess_text(sent.text, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords)
                lst_tokens = nltk.tokenize.word_tokenize(text_clean)
                if(len(lst_tokens) > 4):
                    sentences.append(sent.text)
                    sentences_clean.append(text_clean)
                    listing_ids.append(listing_id)

        corpus_df = pd.DataFrame()
        corpus_df['listing_id'] = listing_ids
        corpus_df['text'] = sentences
        corpus_df['text_clean'] = sentences_clean
        corpus_df["word_count"] = corpus_df['text_clean'].apply(lambda x: len(str(x).split()))

        st.write(corpus_df.shape)
        st.write(corpus_df.head())
        st.write(corpus_df.text_clean[0])

        corpus_buffer = StringIO()
        corpus_df.to_csv(corpus_buffer, index=False)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=cf.S3_DATA_BOOKING + city + '/corpus_df.csv',
            Body = corpus_buffer.getvalue()

        )


        tokenized_corpus = []
        for words in corpus_df['text_clean']:
            tokenized_corpus.append(words.split())

        #pretrained_model = joblib.load(cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        cf.S3_CLIENT.download_file(cf.S3_DATA_PATH, cf.S3_DATA_BOOKING + 'booking_pretrained_model.pkl', cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        pretrained_model = joblib.load(cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        
        st.write(pretrained_model.wv.similarity("bistro","restaurant"))
        st.write(pretrained_model.wv.similarity("train","bus"))
        st.write(pretrained_model.wv.similarity("train","travel"))
        st.write(pretrained_model.wv.similarity("steakhouse","restaurant"))

        pretrained_model.train(tokenized_corpus, total_examples=pretrained_model.corpus_count, epochs = 5)
        joblib.dump(pretrained_model, cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')

        embeddings =  self.vectorize(corpus_df['text_clean'], pretrained_model)
        embeddings_df = pd.DataFrame(embeddings)
        st.write(embeddings_df.shape)
        st.write(embeddings_df.head())    

        scaler = MinMaxScaler()
        embeddings_df = pd.DataFrame(scaler.fit_transform(embeddings_df), columns = range(0,300))
        st.write(embeddings_df.head())   

        embedding_buffer = StringIO()
        embeddings_df.to_csv(embedding_buffer, index=False)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=cf.S3_DATA_BOOKING + city + '/embeddings_df.csv',
            Body = embedding_buffer.getvalue()

        )
        


    ##############################################################################################
    # Knowledge Graph
    ##############################################################################################
    
    def subtree_matcher(self, doc): 
        subj = []
        obj = []
        pron = []
        adj = []
      
        # iterate through all the tokens in the input sentence 
        for i,tok in enumerate(doc): 
        #print(tok.text,tok.dep_, tok.pos_)

            # extract subject 
            if (tok.dep_.endswith("subj") == True) & (tok.pos_.endswith("NOUN") == True):
              subj.append(tok.text)

            # extract subject 
            if (tok.dep_.endswith("ROOT") == True) & (tok.pos_.endswith("PROPN") == True):
              subj.append(tok.text)

            # extract subject 
            if (tok.dep_.endswith("ROOT") == True) & (tok.pos_.endswith("NOUN") == True):
              subj.append(tok.text)

            if (tok.dep_.endswith("subj") == True) & (tok.pos_.endswith("PROPN") == True):
              subj.append(tok.text)

            if (tok.dep_.endswith("compound") == True) & (tok.pos_.endswith("PROPN") == True):
              obj.append(tok.text)
            
            if (tok.dep_.endswith("compound") == True) & (tok.pos_.endswith("NOUN") == True):
              obj.append(tok.text)

            # extract object 
            if (tok.dep_.endswith("obj")) & (tok.pos_.endswith("NOUN") == True): 
              obj.append(tok.text)
            
            # extract object 
            if (tok.dep_.endswith("obj")) & (tok.pos_.endswith("PROPN") == True): 
              obj.append(tok.text)

            # extract object 
            if (tok.dep_.endswith("conj")) & (tok.pos_.endswith("NOUN") == True): 
              obj.append(tok.text)

            # extract object b
            if tok.dep_.endswith("attr") == True: 
              obj.append(tok.text)

            # extract pron 
            if (tok.dep_.endswith("subj") == True) & (tok.pos_.endswith("PRON") == True): 
              pron.append(tok.text)

            if (tok.dep_.endswith("subj") == True) & (tok.pos_.endswith("PROPN") == True):
              pron.append(tok.text)

            if (tok.pos_ == 'ADJ'):
              adj.append(tok.text)
          
        #print(subj)  

        return subj, obj, pron, adj


    
    def extract_info(self, sent):

        doc = nlp(sent)
        clause_pos = [0]
        clause_text = []
        words_sentence = []
        obj1 = []
        obj2 = []

        i = 0
        for token in doc:
            words_sentence.append(token.text)
            #if(doc[i].pos_== "PUNCT"):
            if(token.text == ","):
              clause_pos.append(i + 1)
            i = i + 1
        clause_pos.append(i)

        for i in range(0, len(clause_pos)-1):
            start = clause_pos[i]
            end = clause_pos[i+1] - 1
            clause = ' '.join(words_sentence[start:end+1])
            #print(clause)
            doc = nlp(clause)
            subj, obj, pron, adj = self.subtree_matcher(nlp(clause))
            '''
            print(subj)
            print(obj)
            print(adj)
            print()
            '''
        
        if(len(subj) > 0):
            for se in subj:
                if(len(obj) > 0):
                    for oe in obj:
                        obj1.append(se)
                        obj2.append(oe)
                if(len(adj) > 0):
                    for ae in adj:
                        obj1.append(se)
                        obj2.append(ae)
                if((len(obj) == 0) & (len(adj) == 0)):
                    obj1.append(se)
                    obj2.append('NA')
        
        if(len(subj) == 0):
            if(len(obj) >= 0):
                if(len(adj) >= 0):
                    for oe in obj:
                        for ae in adj:
                                obj1.append(oe)
                                obj2.append(ae)
        
        return obj1, obj2


    
    def generate_graph_data(self, city, listing_id, cluster_id):
        #filename = 'airbnb_cluster' + str(cluster_id) + '.csv'
        filename = city + '/booking_cluster_df.csv'
        house_df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_BOOKING + filename)
        house_df = house_df[house_df.listing_id == listing_id].reset_index(drop=True)
        house_df = house_df[house_df.Cluster == cluster_id].reset_index(drop=True)
        st.write(house_df.shape)
        st.write(house_df.head())

        house_obj1 = []
        house_obj2 = []

        for i in range(0, len(house_df)):
            sent = house_df.text[i]
            #sent = utils_preprocess_text(sent)
            obj1, obj2 = self.extract_info(sent)
            house_obj1 = house_obj1 + obj1
            house_obj2 = house_obj2 + obj2

            graph_df = pd.DataFrame()
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
            
        return graph_df

    
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

        # Save and read graph as HTML file (on Streamlit Sharing)
        html_file = os.path.join(cf.IMAGE_PATH, "pyvis_graph.html")
        try:
            drug_net.save_graph(html_file)
            HtmlFile = open(html_file, 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            drug_net.save_graph(html_file)
            HtmlFile = open(html_file, 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=1000)


    def generate_graph_booking(self, city, listing_id, cluster_id, entities):
        
        graph_df = self.generate_graph_data(city, listing_id, cluster_id)
        entity_list = self.extract_entities(graph_df)
        if(len(entities) == 0):
            entities = entity_list
            flag = 1
        else:
            graph_df = graph_df[(graph_df.obj1.isin(st.session_state.ENTITY_OPTION)) | (graph_df.obj2.isin(st.session_state.ENTITY_OPTION))].reset_index()
            flag = 0
        weight_graph_df = pd.DataFrame(graph_df.groupby(['obj1','obj2']).count()).reset_index()
        weight_graph_df.rename(columns={'ID':'weight'}, inplace=True)
        weight_graph_df = weight_graph_df.drop_duplicates(subset=['obj1','obj2'], keep='last')

        # Generate a networkx graph based on subset data
        net_repulsion = self.generate_network_viz(weight_graph_df, 'obj1','obj2', 'weight', layout='repulsion')


        return entities, flag


    ##############################################################################################
    # Dimensionality Reduction
    ##############################################################################################
    def train_autoencoder(self, city, encoding1_dim=80, encoding2_dim=30, latent_dim=15, n_clusters=7):

        X_train = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/embeddings_df.csv')
        st.write(X_train.head())

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

        nb_epoch = 20
        autoencoder.compile(optimizer='adam', loss='mse')

        cp = ModelCheckpoint(filepath='autoencoder1.h5', save_best_only=True, verbose=0)
        tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        t_ini = datetime.datetime.now()
        history = autoencoder.fit(X_train, 
                                  X_train, 
                                  epochs=nb_epoch, 
                                  #batch_size=batch_size, 
                                  shuffle=True, 
                                  validation_split=0.1, 
                                  verbose=1
        ).history
        t_fin = datetime.datetime.now()
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
        ae_embeddings = pd.DataFrame(encoded)
        st.write(ae_embeddings.shape)
        st.write(ae_embeddings.head())

        keras.models.save_model(autoencoder, city + "_booking_autoencoder_model.h5")


        kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        kmeans.fit(ae_embeddings)
        cluster_labels = kmeans.labels_
        cluster_df = pd.DataFrame()
        cluster_df['KMeans_cluster'] = cluster_labels

        tsne = TSNE(n_components=2, random_state=9)
        X_train_tsne = tsne.fit_transform(ae_embeddings)
        X_train_tsne = pd.DataFrame(X_train_tsne)
        X_train_tsne['KMeans_cluster'] = cluster_df['KMeans_cluster']
        plot_cluster(X_train_tsne[[0,1]].values, X_train_tsne['KMeans_cluster'], n_clusters)


        corpus_df = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/corpus_df.csv')
        corpus_df['Cluster'] = cluster_df['KMeans_cluster']
        ae_embeddings['Cluster'] = cluster_df['KMeans_cluster']

        self.show_cluster(corpus_df, n_clusters)

        corpus_buffer = StringIO()
        corpus_df.to_csv(corpus_buffer)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=cf.S3_DATA_BOOKING + city + '/booking_cluster_df.csv',
            Body = corpus_buffer.getvalue()

        )  

        ae_embedding_buffer = StringIO()
        ae_embeddings.to_csv(ae_embedding_buffer)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=cf.S3_DATA_BOOKING + city + '/booking_ae_embedding_df.csv',
            Body = ae_embedding_buffer.getvalue()

        )  



    def topic_modeling_booking(self):

        ae_embeddings = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/booking_ae_embedding_df.csv')

        kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        kmeans.fit(ae_embeddings)
        cluster_labels = kmeans.labels_
        cluster_df = pd.DataFrame()
        cluster_df['KMeans_cluster'] = cluster_labels

        tsne = TSNE(n_components=2, random_state=9)
        X_train_tsne = tsne.fit_transform(ae_embeddings)
        X_train_tsne = pd.DataFrame(X_train_tsne)
        X_train_tsne['KMeans_cluster'] = cluster_df['KMeans_cluster']
        plot_cluster(X_train_tsne[[0,1]].values, X_train_tsne['KMeans_cluster'], n_clusters)


        corpus_df = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/corpus_df.csv')
        corpus_df['Cluster'] = cluster_df['KMeans_cluster']
        ae_embeddings['Cluster'] = cluster_df['KMeans_cluster']

        self.show_cluster(corpus_df, n_clusters)

        corpus_buffer = StringIO()
        corpus_df.to_csv(corpus_buffer)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=cf.S3_DATA_BOOKING + city + '/booking_cluster_df.csv',
            Body = corpus_buffer.getvalue()

        )        


    def topic_modeling_airbnb(self):
        ae_embeddings = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'airbnb_ae_embeddings.csv')
        st.write(ae_embeddings.head())
        X_train_tsne = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'X_train_tsne_with_cluster.csv')
        plot_cluster(X_train_tsne[['0','1']].values, X_train_tsne['KMeans_cluster'], 7) 

        corpus_cluster_df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'airbnb_cluster_df.csv')
        self.show_cluster(corpus_cluster_df, 7)




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


    def show_cluster(self, corpus_cluster_df, n_clusters):
        st.write(corpus_cluster_df.head())
        n_records = 5
        for i in range(0, n_clusters):          
            st.write(i)
            for j in range(0, n_records):
                st.write(corpus_cluster_df[(corpus_cluster_df.Cluster == i)].text.iloc[j])
                st.write()
        

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
        embedded_value = encoder_layer.predict(embedded_value)
        cosine_sim = cosine_similarity(embeddings,embedded_value)
        cluster_df['Cosine'] = cosine_sim
        cluster_df = cluster_df.sort_values(['listing_id', 'Cosine'], ascending=False)

        return cluster_df


    def avg_similarity(self, similarity_df, topic):
        sim_df = similarity_df.groupby(['listing_id']).Cosine.max()
        sim_df = pd.DataFrame(sim_df).reset_index(drop=False)
        sim_df.columns = ['listing_id', 'score' + str(topic)]
        sim_df['count'+ str(topic)] = similarity_df.groupby(['listing_id']).listing_id.count().values
        #sim_df = sim_df.sort_values('score', ascending=False)
        return sim_df




    def hotel_recommendation_booking(self, city, 
                                    cluster0='', 
                                    cluster1='', 
                                    cluster2='', 
                                    cluster3='', 
                                    cluster4='', 
                                    cluster5='', 
                                    cluster6=''):

        train_vars = range(0,15)
        train_vars = [str(var) for var in train_vars]

        corpus_df = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/corpus_df.csv')
        pretrained_model = joblib.load(cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        #cf.S3_CLIENT.download_file(cf.S3_DATA_PATH, cf.S3_DATA_BOOKING + 'booking_pretrained_model.pkl', cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        #pretrained_model = joblib.load(cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        ae_embeddings = dm.s3_load_csv(cf.S3_DATA_PATH,  cf.S3_DATA_BOOKING + city + '/booking_ae_embedding_df.csv')

        autoencoder = keras.models.load_model(city + "_booking_autoencoder_model.h5")
        encoder_layer = Model(autoencoder.input, autoencoder.layers[-4].output)

        df0 = self.calculate_similarity(cluster0, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        df1 = self.calculate_similarity(cluster1, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        df2 = self.calculate_similarity(cluster2, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        df3 = self.calculate_similarity(cluster3, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        df4 = self.calculate_similarity(cluster4, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        df5 = self.calculate_similarity(cluster5, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)
        df6 = self.calculate_similarity(cluster6, pretrained_model, ae_embeddings[train_vars], corpus_df, encoder_layer)

        sim0_df = self.avg_similarity(df0, 0)
        sim1_df = self.avg_similarity(df1, 1)
        sim2_df = self.avg_similarity(df2, 2)
        sim3_df = self.avg_similarity(df3, 3)
        sim4_df = self.avg_similarity(df4, 4)
        sim5_df = self.avg_similarity(df5, 5)
        sim6_df = self.avg_similarity(df6, 6)

        sum_sim_df = pd.DataFrame(corpus_df['listing_id'].unique(), columns =['listing_id'])
        sum_sim_df = sum_sim_df.merge(sim0_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim1_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim2_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim3_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim4_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim5_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim6_df, how='left', left_on='listing_id', right_on='listing_id')

        sum_sim_df.fillna(0, inplace=True)
        sum_sim_df['avg_sim'] = sum_sim_df[['score0', 'score1', 'score2', 'score3', 'score4','score5','score6']].mean(axis=1)
        sum_sim_df['avg_count'] = sum_sim_df[['count0', 'count1', 'count2', 'count3', 'count4','count5','count6']].mean(axis=1)
        sum_sim_df['total_score'] = sum_sim_df['avg_sim'] * sum_sim_df['avg_count']
        sum_sim_df = sum_sim_df.sort_values(['avg_sim'], ascending=False)
        st.write(sum_sim_df.shape)
        st.write(sum_sim_df.head(20))




    def hotel_recommendation(self, cluster0='', 
                                   cluster1='', 
                                   cluster2='', 
                                   cluster3='', 
                                   cluster4='', 
                                   cluster5='', 
                                   cluster6=''):

        train_vars = range(0,15)
        train_vars = [str(var) for var in train_vars]

        # overall experience/thought
        cluster0 = 'a great place to stay for a trip to London, recommend and encourage any tourists, good value in a place to stay while exploring London'

        # breakfast
        cluster1 = 'breakfast included coffee, milk, cereal, bread, ham, cheese, fruit, juice and a British tea selection'

        # room with amenities
        cluster2 = 'tv, hair dryer, kettle, microwave, cups, dishes, cutlery, cereals, shower, usb friendly charger ports, regular charger ports, house key, towels and flannels, desk, robes'

        # neighborhood and attraction
        cluster3 = 'neighbourhood filled with shops, restaurants, pubs, park'

        #host
        cluster4 = 'pleasant host, fast response, lots of advice to visit the city easily and quickly, transport links to where to eat locally, checked in on us to make sure we are doing alright'

        # room/flat in general conditions
        cluster5 = 'The room is clean, spacious, beautiful, furnished, full of litte details, the beds are comfy, the bathroom is clean,  garden and trees'

        # transportation
        cluster6 = 'easy access to Central London, area with easy bus, tube, underground and train connections to London Bridge, Victoria Station,Canary Wharf, london tower'


        df_train = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'airbnb_cluster_df.csv')
        #pretrained_model = joblib.load(cf.TRAINED_MODEL_PATH + '/airbnb_pretrained_model.pkl')
        #cf.S3_CLIENT.download_file(cf.S3_DATA_PATH, cf.S3_DATA_BOOKING + 'booking_pretrained_model.pkl', cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        #pretrained_model = joblib.load(cf.TRAINED_MODEL_PATH + '/booking_pretrained_model.pkl')
        ae_embeddings = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'airbnb_ae_embeddings.csv')

        autoencoder = keras.models.load_model("airbnb_autoencoder_model.h5")
        encoder_layer = Model(autoencoder.input, autoencoder.layers[-4].output)

        df0 = self.calculate_similarity(cluster0, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)
        df1 = self.calculate_similarity(cluster1, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)
        df2 = self.calculate_similarity(cluster2, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)
        df3 = self.calculate_similarity(cluster3, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)
        df4 = self.calculate_similarity(cluster4, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)
        df5 = self.calculate_similarity(cluster5, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)
        df6 = self.calculate_similarity(cluster6, pretrained_model, ae_embeddings[train_vars], df_train, encoder_layer)

        sim0_df = self.avg_similarity(df0, 0)
        sim1_df = self.avg_similarity(df1, 1)
        sim2_df = self.avg_similarity(df2, 2)
        sim3_df = self.avg_similarity(df3, 3)
        sim4_df = self.avg_similarity(df4, 4)
        sim5_df = self.avg_similarity(df5, 5)
        sim6_df = self.avg_similarity(df6, 6)

        sum_sim_df = pd.DataFrame(df_train['listing_id'].unique(), columns =['listing_id'])
        sum_sim_df = sum_sim_df.merge(sim0_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim1_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim2_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim3_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim4_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim5_df, how='left', left_on='listing_id', right_on='listing_id')
        sum_sim_df = sum_sim_df.merge(sim6_df, how='left', left_on='listing_id', right_on='listing_id')

        sum_sim_df.fillna(0, inplace=True)
        sum_sim_df['avg_sim'] = sum_sim_df[['score0', 'score1', 'score2', 'score3', 'score4','score5','score6']].mean(axis=1)
        sum_sim_df['avg_count'] = sum_sim_df[['count0', 'count1', 'count2', 'count3', 'count4','count5','count6']].mean(axis=1)
        sum_sim_df['total_score'] = sum_sim_df['avg_sim'] * sum_sim_df['avg_count']
        sum_sim_df = sum_sim_df.sort_values(['avg_sim'], ascending=False)
        st.write(sum_sim_df.shape)
        st.write(sum_sim_df.head(20))

