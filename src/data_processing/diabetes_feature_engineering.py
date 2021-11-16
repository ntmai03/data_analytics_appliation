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


diabetes_median_imputer = os.path.join(cf.PIPELINE_PATH, 'diabetes_median_imputer.npy')
diabetes_iterative_imputer = os.path.join(cf.PIPELINE_PATH, 'diabetes_iterative_imputer.npy')
diabetes_scaler = os.path.join(cf.PIPELINE_PATH, 'diabetes_scaler.pkl')


def fill_numerical_na(df, var_list, train_flag=0):
    
    data = df.copy()
    
    if(train_flag == 1):
        median_var_dict = {}
        # add variable indicating missingess + median imputation
        for var in var_list:
            median_val = data[var].median()
            median_var_dict[var] = median_val    
        # save result
        np.save(diabetes_median_imputer, median_var_dict)
    else:
        median_var_dict = np.load(diabetes_median_imputer, allow_pickle=True).item()
    
    for var in var_list:
        median_val = median_var_dict[var]
        data[var].fillna(median_val, inplace=True)
    
    return data



def iterative_imputer(df, var_list, train_flag=0):
    
    data = df.copy()   
      
    imputer = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
    
    if(train_flag == 1):
        imputer.fit(data[var_list])
        joblib.dump(imputer, diabetes_iterative_imputer)
    else:
        imputer = joblib.load(diabetes_iterative_imputer)  
        
    data[var_list] = imputer.transform(data[var_list])    
    
    return data


def scaling_data(df, var_list, train_flag=0):
    
    data = df.copy()

    # fit scaler
    scaler = MinMaxScaler() # create an instance
    scaler.fit(data[var_list]) #  fit  the scaler to the train set for later use
    
    # we persist the model for future use
    if(train_flag == 1):
        joblib.dump(scaler, diabetes_scaler)
    scaler = joblib.load(diabetes_scaler)  
    
    data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)
    
    return data