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

import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

import spacy
#spacy.download('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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


    
    def generate_graph_data(self, listing_id, cluster_id):
        house_df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'airbnb_cluster_df.csv')
        st.write('step 1')
        st.write(house_df.head())
        house_df = house_df[house_df.listing_id == listing_id]
        cluster_df = house_df[house_df.Cluster == cluster_id].reset_index()
        st.write('step 2')
        st.write(cluster_df.shape)

        house_obj1 = []
        house_obj2 = []

        for i in range(0, len(cluster_df)):
            sent = cluster_df.text[i]
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


    
    def generate_graph(self, listing_id = 1296836, cluster_id = 6):
        
        
        
        graph_df = self.generate_graph_data(listing_id, cluster_id)
        entity_list = self.extract_entities(graph_df)
        graph_df = graph_df[(graph_df.obj1.isin(entity_list)) | (graph_df.obj2.isin(entity_list))].reset_index()

        weight_graph_df = pd.DataFrame(graph_df.groupby(['obj1','obj2']).count()).reset_index()
        weight_graph_df.rename(columns={'ID':'weight'}, inplace=True)
        weight_graph_df = weight_graph_df.drop_duplicates(subset=['obj1','obj2'], keep='last')

        # Generate a networkx graph based on subset data
        net_repulsion = self.generate_network_viz(weight_graph_df, 'obj1','obj2', 'weight', layout='repulsion')



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


        

