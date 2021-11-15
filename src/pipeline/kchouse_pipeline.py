# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
from pathlib import Path
import os

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd
from math import sqrt

# Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# statsmodels
import math

# to persist the model and the scaler
import joblib

from src.data_processing import kchouse_feature_engineering as fe
from src.util import data_manager as dm
from src import config as cf

# variables
#FEATURES = list(df.columns)

# rename columns
FEATURE_MAP = {'date': 'date',
                'price': 'price'}

# data type conversion
DATA_TYPE = {'zipcode': 'str',
             'date': 'object',
             'price': 'float64',
             'bedrooms': 'int64',
             'bathrooms': 'int64',
             'sqft_living': 'int64',
             'sqft_lot': 'int64',
             'floors': 'int64',
             'waterfront': 'int64',
             'view': 'int64',
             'condition': 'int64',
             'grade': 'int64',
             'sqft_above': 'int64',
             'sqft_basement': 'int64',
             'yr_built': 'int64',
             'yr_renovated': 'int64',
             'lat': 'float64',
             'long': 'float64',
             'sqft_living15': 'int64',
             'sqft_lot15': 'int64'}

TARGET = 'price'

TEMPORAL_VARS = ['year']

TEXT_VARS = []

# categorical variables to encode
#CATEGORICAL_VARS = [var for var in df.columns if df[var].dtypes == 'O' if var not in TARGET + TEXT_VARS + TEMPORAL_VARS]
TEMP_CATEGORICAL_VARS = ['zipcode']

CATEGORICAL_VARS = ['season']

#NUMERICAL_VARS = [var for var in df.columns if df[var].dtypes != 'O']
TEMP_NUMERICAL_VARS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                  'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                  'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                  'sqft_living15', 'sqft_lot15']

NUMERICAL_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                  'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                  'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                  'sqft_living15', 'sqft_lot15', 'sqft_ratio', 'zipcode']

DUMMY_VARS = []

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = []

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = []

# variables to log transform
NUMERICALS_LOG_VARS = []

# drop features
DROP_FEATURES = ['date']

def clean_data(df):
    
    data = df.copy()
    
    # Rename columns
    data.rename(columns=FEATURE_MAP, inplace=True)
    
    # data type conversion
    for key in DATA_TYPE:
        data[key] = data[key].astype(DATA_TYPE[key])
    
    # Remove duplicated data
    data = data.drop_duplicates(keep = 'last')
    
    # Reset index
    data = data.reset_index(drop = True)
    
    return data

def data_engineering_pipeline1(df, train_flag=0):

    print('train_flag:', train_flag)
     
    df = clean_data(df)
    df = fe.date_transformer(df, 'date')
    df = fe.sqft_ratio(df, 'sqft_living', 'sqft_living15')
    df = fe.encode_categorical(df, TEMP_CATEGORICAL_VARS, TARGET, train_flag)
    data_categorical = fe.create_dummy_vars(df, CATEGORICAL_VARS, DUMMY_VARS, train_flag)
    data_scale = fe.scaling_data(df, NUMERICAL_VARS, train_flag)
    
    df = pd.concat([data_scale,data_categorical], axis=1)

    return df



"""
import xgboost as xgb

data_file = os.path.join(cf.DATA_RAW_PATH, "kc_house_data.csv")
df = dm.load_csv_data(data_file)


X_train, X_test, y_train, y_test = dm.split_data(df, df[TARGET])
new_obj = pd.DataFrame(X_test.iloc[0]).T

processed_X_train = data_engineering_pipeline1(X_train, train_flag=1)
TRAIN_VARS = list(processed_X_train.columns)

# transform the target
log_y_train = np.log(y_train)

xgb_model = xgb.XGBRegressor()
xgb_model.fit(processed_X_train, log_y_train)

model_file_name = 'kchouse_xgb.pkl'
save_path = os.path.join(cf.TRAINED_MODEL_PATH, model_file_name)
joblib.dump(xgb_model, save_path)
"""

  
