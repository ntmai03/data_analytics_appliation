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


diabetes_median_imputer = os.path.join(cf.PIPELINE_PATH, 'credit_risk_median_imputer.npy')
diabetes_iterative_imputer = os.path.join(cf.PIPELINE_PATH, 'credit_risk_iterative_imputer.npy')
diabetes_scaler = os.path.join(cf.PIPELINE_PATH, 'credit_risk_scaler.pkl')
ordinal_label_path = os.path.join(cf.PIPELINE_PATH, 'credit_risk_OrdinalLabels.npy')

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


def calculate_class_ratio(data, var, target):
    class_0 = data[data[target]==0].groupby(var).count()[target]
    class_1 = data[data[target]==1].groupby(var).count()[target]
    class_ratio = class_1/class_0   
    return  class_ratio

def encode_categorical(df, var_list, target, train_flag=0):
    
    data = df.copy()

    if(train_flag == 1):
        ordinal_label_dict = {}
        for var in var_list:
            st.write(var)
            ordinal_label = calculate_class_ratio(data, var, target)
            ordinal_label_dict[var] = ordinal_label

        # now we save the dictionary
        np.save(ordinal_label_path, ordinal_label_dict)
    else:
        ordinal_label_dict = np.load(ordinal_label_path, allow_pickle=True).item()
        
    for var in var_list:
        ordinal_label = ordinal_label_dict[var]
        ordinal_label_df = pd.DataFrame(ordinal_label).reset_index(drop=False)
        ordinal_label_df.columns = [var, var + '_ratio']
        ordinal_label_df.ratio = ordinal_label_df[var + '_ratio'].replace(np.nan, 0)
        data = data.merge(ordinal_label_df, how='left', left_on = var, right_on=var)
    
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


def create_num_of_days(df):
    
    data = df.copy()

    data['issue_d_date'] = pd.to_datetime(data['issue_d'], format = '%b-%y')
    data['earliest_cr_line_date'] = pd.to_datetime(data['earliest_cr_line'], format = '%b-%y')
    data['num_of_days'] = data['issue_d_date'] - data['earliest_cr_line_date']
    data['num_of_days'] = data['num_of_days'].apply(lambda x: x.days)
    
    return data


def create_num_emp_length(df):

    data = df.copy()

    data['num_emp_length'] = data['emp_length'].map({'10+ years':10, 
                                         '9 years':9, 
                                         '8 years':8,
                                         '7 years':7,
                                         '6 years':6,
                                         '5 years':5,
                                         '4 years':4,
                                         '3 years':3,
                                         '2 years':2,
                                         '1 year':1,
                                         '< 1 year':0.5})    
    return data