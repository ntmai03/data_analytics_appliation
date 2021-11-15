# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
from pathlib import Path
import os
import streamlit as st

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd
from math import sqrt

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

sys.path.append('src')

# to persist the model and the scaler
import joblib
import config as cf


dummy_path = os.path.join(cf.PIPELINE_PATH, 'kchouse_train_dummy.npy')
ordinal_label_path = os.path.join(cf.PIPELINE_PATH, 'kchouse_OrdinalLabels.npy')
scaler_path = os.path.join(cf.PIPELINE_PATH, 'kchouse_scaler.pkl')


def date_transformer(df, date):
    
    data = df.copy()
    
    data[date] = pd.to_datetime(data[date])
    data['month'] = data[date].apply(lambda date:date.month)
    data['year'] = data[date].apply(lambda date:date.year)
    data['season'] = 'NA'
    data.loc[data.month.isin([12,1,2]), 'season'] = 'winter'
    data.loc[data.month.isin([3,4,5]), 'season'] = 'spring'
    data.loc[data.month.isin([6,7,8]), 'season'] = 'summer'
    data.loc[data.month.isin([9,10, 11]), 'season'] = 'autum'
    
    return data


def sqft_ratio(df, var1, var2):
    
    data = df.copy()
    
    data['sqft_ratio'] = data[var1]/data[var2]
    
    return data


def fill_numerical_na(df, var_list, train_flag=0):
    
    data = df.copy()
    
    if(train_flag == 1):
        mean_var_dict = {}
        # add variable indicating missingess + median imputation
        for var in var_list:
            mean_val = X_train[var].median()
            mean_var_dict[var] = mean_val    
        # save result
        np.save('kchouse_mean_var_dict.npy', mean_var_dict)
    else:
        mean_var_dict = np.load('kchouse_mean_var_dict.npy', allow_pickle=True).item()
    
    for var in var_list:
        mean_val = mean_var_dict[var]
        data[var+'_NA'] = np.where(data[var].isnull(), 1, 0)
        data[var].fillna(mean_val, inplace=True)
    
    return data


# function to replace NA in categorical variables
def fill_categorical_na(df, var_list):
    
    data = df.copy()
    
    data[var_list] = data[var_list].fillna('Missing')
    
    return data

def log_transform(df, var_list):
    
    data = df.copy()
    
    for var in var_list:
        if(len(data[var]) > 1):
            data[var] = np.log(data[var])
        else:
            data[var] = np.log(data[var][0])
    
    return data

def find_frequent_labels(data, var, rare_perc):
    # finds the labels that are shared by more than a certain % of the houses in the dataset
    tmp = data.groupby(var)[TARGET].count() / len(data)
    return tmp[tmp>rare_perc].index


def replace_rare_values(df, var_list, train_flag=0):
    
    data = df.copy()
    
    if(train_flag == 1):
        frequent_labels_dict = {}
        for var in var_list:
            frequent_ls = find_frequent_labels(data, var, 0.01)   
            # we save the list in a dictionary
            frequent_labels_dict[var] = frequent_ls
        # now we save the dictionary
        np.save('kchouse_FrequentLabels.npy', frequent_labels_dict)
    else:
        frequent_labels_dict = np.load('kchouse_FrequentLabels.npy', allow_pickle=True).item()
        
    for var in var_list:
        frequent_ls = frequent_labels_dict[var]
        data[var] = np.where(data[var].isin(frequent_ls), data[var], 'Rare')
        
    return data


def replace_categories(train, var, target):
    ordered_labels = train.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 
    return ordinal_label    


def encode_categorical(df, var_list, target, train_flag=0):
    
    data = df.copy()

    if(train_flag == 1):
        ordinal_label_dict = {}
        for var in var_list:
            ordinal_label = replace_categories(data, var, target)
            ordinal_label_dict[var] = ordinal_label
        # now we save the dictionary
        np.save(ordinal_label_path, ordinal_label_dict)
    else:
        ordinal_label_dict = np.load(ordinal_label_path, allow_pickle=True).item()
        
    for var in var_list:
        ordinal_label = ordinal_label_dict[var]
        data[var] = data[var].map(ordinal_label)
    
    return data


def create_dummy_vars(df, var_list, DUMMY_VARS, train_flag=0):  
    
    data = df.copy()
    data_categorical = pd.DataFrame()
    for var in var_list:
        data_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep='_',drop_first=True)  
        data_categorical = pd.concat([data_categorical, data_dummies], axis=1)    
    
    if(train_flag == 1):
        train_dummy = list(data_categorical.columns)
        pd.Series(train_dummy).to_csv(dummy_path, index=False)
    else:
        test_dummy = list(data_categorical.columns)
        train_dummy = pd.read_csv(dummy_path)
        train_dummy.columns = ['Name']
        train_dummy = list(train_dummy.Name.values)   
        
    for col in train_dummy:
        if col not in data_categorical:
            data_categorical[col] = 0
    if(len(DUMMY_VARS) > 0):
        data_categorical = data_categorical[DUMMY_VARS] 
    
    return data_categorical


def scaling_data(df, var_list, train_flag=0):
    
    data = df.copy()
   
    # fit scaler
    scaler = MinMaxScaler() # create an instance
    scaler.fit(data[var_list]) #  fit  the scaler to the train set for later use
    
    # we persist the model for future use
    if(train_flag == 1):
        joblib.dump(scaler, scaler_path)
    scaler = joblib.load(scaler_path)  
    
    data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)
    
    return data


