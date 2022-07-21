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


shopping_idx2uid = os.path.join(cf.ANALYSIS_PATH, 'shopping_idx2uid.npy')
shopping_uid2idx = os.path.join(cf.ANALYSIS_PATH, 'shopping_uid2idx.npy')
shopping_idx2iid = os.path.join(cf.ANALYSIS_PATH, 'shopping_idx2iid.npy')
shopping_iid2idx = os.path.join(cf.ANALYSIS_PATH, 'shopping_iid2idx.npy')


class ShoppingRecommendation:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """


    def __init__(self):
 
        self.transaction_detail = None
        self.transaction_data = None
        self.item_data = None
        self.user_data = None
        self.idx2uid = None
        self.uid2idx = None
        self.idx2iid = None
        self.iid2idx = None
        self.user_item_mt = None
        self.item_user_mt = None
        self.df_train = None
        self.df_test = None

        self.user_id = 'CustomerID'
        self.item_id = 'ItemID'
        self.user_index = 'UserIndex'
        self.item_index = 'ItemIndex'
        self.value = 'Quantity'

    ##############################################################################################
    # Data Processing
    ##############################################################################################
    def load_dataset(self):

        '''
        # get data from local machine
        data_file = os.path.join(cf.DATA_RAW_PATH, "diabetes.csv")
        self.data = dm.load_csv_data(data_file)
        '''

        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "kc_house_data.csv")


    def prepare_dataset(self):

        # get data from s3
        transaction_detail = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'shopping_transaction_detail.csv')
        item_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'shopping_item_data.csv')
        user_data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'shopping_user_data.csv')
        idx2uid = np.load(shopping_idx2uid, allow_pickle=True).item()
        uid2idx = np.load(shopping_uid2idx, allow_pickle=True).item()
        idx2iid = np.load(shopping_idx2iid, allow_pickle=True).item()
        iid2idx = np.load(shopping_iid2idx, allow_pickle=True).item()

        # split dataset
        train_data, test_data = train_test_split(self.transaction_detail, test_size=test_size)

        # create user-item matrix
        item_user_mt = train_data.pivot_table(index=item_index, columns=user_index, values=value)
        item_user_mt.fillna(0, inplace=True)


        self.transaction_detail = transaction_detail
        self.item_data = item_data
        self.user_data = user_data
        self.idx2uid = idx2uid
        self.uid2idx = uid2idx
        self.idx2iid = idx2iid
        self.iid2idx = iid2idx
        self.df_train = train_data
        self.df_test = test_data
        self.user_item_mt = user_item_mt
        self.item_user_mt = user_item_mt.T

        


        


    ##############################################################################################
    # Predictive Model
    ##############################################################################################
 
    def Train_ItemItemCF(self, ItemID):

        self.prepare_dataset()

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
        st.write(item_simdf.sort_values('sim_score', ascending=False).head(5))

        # store file to S3



    def Recommend_ItemItemCF(self, UserID):
        self.prepare_dataset()
        # get all items of a specific user
        user_items = list(self.df_train[df_train[user_id] == UserID][item_index].unique())
        # get all items in the system
        all_items = list(self.df_train[item_index].unique())
        # create cooccurence matrix to keep track the common items of a given user and the remaining users
        cooccurence_matrix = []

        # for each item i in user's item list, find similar items with this item
        for item_i in user_items:
            st.write('ItemID = ', str(item_i))
            item_i_user = self.item_user_mt.loc[item_user_mt.index == item_i,:]
            item_simdf = pd.DataFrame(cosine_similarity(item_user_mt, item_user_array))
            item_simdf.columns = ['sim_score']
            item_simdf[self.item_index] = item_simdf.index.values
            item_simdf[self.item_id] = item_simdf.apply(lambda x:self.idx2iid[x[self.item_index]], axis=1)
            cooccurence_matrix.append(jac_sim[:,0].tolist())
        cooccurence_matrix = pd.DataFrame(cooccurence_matrix, columns=item_user_mt.index.values)
        st.write(cooccurence_matrix.head(3))





