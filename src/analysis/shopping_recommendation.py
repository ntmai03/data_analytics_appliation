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


import time
import datetime
from datetime import datetime
from importlib import reload
from collections import Counter
from sklearn.utils import shuffle

import sklearn
from scipy.sparse import csc_matrix
#from sparsesvd import sparsesvd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error

# metrics to calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

import joblib

from src.util import data_manager as dm
from src.util import regression_util as regu
from src import config as cf


# Evaluation metrics for Regression


house_price_encode_ordinal_label = os.path.join(cf.ANALYSIS_PATH, 'house_price_encode_ordinal_label.npy')
house_price_median_imputer = os.path.join(cf.ANALYSIS_PATH, 'house_price_median_imputer.npy')
house_price_knn_imputer = os.path.join(cf.ANALYSIS_PATH, 'house_price_knn_imputer.npy')
house_price_scaler = os.path.join(cf.ANALYSIS_PATH, 'house_price_scaler.pkl')
house_price_dummy_vars = os.path.join(cf.ANALYSIS_PATH, 'house_price_dummy_vars.npy')


class ShoppingRecommendation:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """


    def __init__(self):
 
        self.transaction_detail = None
        self.item_data = None
        self.user_data = None
        self.idx2uid = None
        self.uid2idx = None
        self.idx2iid = None
        self.iid2idx = None

        self.user_id = 'CustomerID'
        self.item_id = 'ItemID'
        self.user_index = 'UserIndex'
        self.item_index = 'ItemIndex'
        self.value = 'Quantity'

    ##############################################################################################
    # Data Processing
    ##############################################################################################
    # def  impute_median_na(self, var_list, train_flag=0):


    def load_dataset(self):

        '''
        # get data from local machine
        data_file = os.path.join(cf.DATA_RAW_PATH, "diabetes.csv")
        self.data = dm.load_csv_data(data_file)
        '''

        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "kc_house_data.csv")
        
        # Split data to train set and test set       
        self.X_train, self.X_test, self.y_train, self.y_test = dm.split_data(self.data, self.data[self.target])


    def prepare_dataset(self):

        # get data from s3
        df_train = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'houseprice_train.csv')
        df_test = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'houseprice_test.csv')
 
        user_id = 'CustomerID'
        item_id = 'ItemID'
        value = 'Quantity'
        
        transaction_detail = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + 'transaction_detail.csv')
        item_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + 'item_data.csv')
        user_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + 'user_data.csv')
        
        # remove invalid rows
        transaction_detail = transaction_detail[transaction_detail[user_id].isna() == False]
        transaction_detail = transaction_detail[transaction_detail[item_id].isna() == False]
        transaction_detail[user_id] = transaction_detail[user_id].astype(int)
        transaction_detail[item_id] = transaction_detail[item_id].astype(int)
        
        # create mapping for userid and itemid
        N = transaction_detail[user_id].nunique()
        M = transaction_detail[item_id].nunique()
        
        # save the original userId and itemId
        uid = transaction_detail[user_id].unique()
        iid = transaction_detail[item_id].unique()

        # create user_index and item_index
        user_index = np.arange(N)
        item_index = np.arange(M)

        # map ids and their indices
        idx2uid = dict(zip(user_index, uid))
        uid2idx = dict(zip(uid, user_index))
        idx2iid = dict(zip(item_index, iid))
        iid2idx = dict(zip(iid, item_index))

        # apply index to dataframe
        transaction_detail['UserIndex'] = transaction_detail.apply(lambda x:uid2idx[x[user_id]], axis=1)
        transaction_detail['ItemIndex'] = transaction_detail.apply(lambda x:iid2idx[x[item_id]], axis=1)

        self.transaction_detail = transaction_detail
        self.item_data = item_data
        self.user_data = user_data
        self.idx2uid = idx2uid
        self.uid2idx = uid2idx
        self.idx2iid = idx2iid
        self.iid2idx = iid2idx
        



    ##############################################################################################
    # Predictive Model
    ##############################################################################################
 
    def ItemItemCF(self, ItemID):

        self.prepare_dataset()
        # split data
        df = shuffle(self.transaction_detail.copy())
        cutoff = int(0.8 * len(df))
        df_train = df.iloc[:cutoff]
        df_test = df.iloc[cutoff:]

        item_user_mt = df_train.pivot_table(index=self.item_index, columns=self.user_index, values=self.value)
        item_user_mt.fillna(0, inplace=True)       

        ItemIndex = self.iid2idx[ItemID]
        item_user_array = item_user_mt[item_user_mt.index==ItemIndex].values
        item_simdf = pd.DataFrame(cosine_similarity(item_user_mt, item_user_array))
        item_simdf.columns = ['sim_score']
        item_simdf[self.item_index] = item_simdf.index.values
        item_simdf[self.item_id] = item_simdf.apply(lambda x:self.idx2iid[x[self.item_index]], axis=1)
        item_simdf = item_simdf.merge(self.item_data, left_on=self.item_id, right_on=self.item_id, how='left')
        st.write(item_simdf.shape)
        st.write(item_simdf.sort_values('sim_score', ascending=False).head())
