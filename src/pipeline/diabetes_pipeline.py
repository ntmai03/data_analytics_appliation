# Python ≥3.5 is required
import sys
from pathlib import Path
import os

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd

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

# to persist the model and the scaler
import joblib

from src.data_processing import diabetes_feature_engineering as fe
from src.util import data_manager as dm
from src import config as cf


# rename columns
FEATURE_MAP = {'Outcome': 'Outcome'}

# data type conversion
DATA_TYPE = {'Pregnancies': 'int64',
             'Glucose': 'int64',
             'BloodPressure': 'int64',
             'SkinThickness': 'int64',
             'Insulin': 'int64',
             'BMI': 'float64',
             'DiabetesPedigreeFunction': 'float64',
             'Age': 'int64',
             'Outcome': 'int64',}

TARGET = 'Outcome'

TEMPORAL_VARS = []

TEXT_VARS = []

CATEGORICAL_VARS = []

NUMERICAL_VARS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

DUMMY_VARS = []

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = []

# variables to log transform
NUMERICALS_LOG_VARS = []

# drop features
DROP_FEATURES = []


def clean_data(df):
    
    data = df.copy()
    
    # Rename columns
    data.rename(columns=FEATURE_MAP, inplace=True)
    
    # data type conversion
    for key in DATA_TYPE:
        data[key] = data[key].astype(DATA_TYPE[key])
    
    # Remove duplicated data
    data = data.drop_duplicates(keep = 'last')
    
    # Replace 0 with NaN value
    data[NUMERICAL_VARS_WITH_NA] = df[NUMERICAL_VARS_WITH_NA].replace(0,np.NaN)
    
    # Reset index
    data = data.reset_index(drop = True)
    
    return data


def data_engineering_pipeline1(df, train_flag=0):
         
    df = clean_data(df) 
    df = fe.fill_numerical_na(df, NUMERICAL_VARS_WITH_NA, train_flag)
    data_scale = fe.scaling_data(df, NUMERICAL_VARS, train_flag)
    df = pd.DataFrame(data_scale, columns = NUMERICAL_VARS)

    return df


def data_engineering_pipeline2(df, train_flag=0):
         
    df = clean_data(df) 
    df = fe.iterative_imputer(df, NUMERICAL_VARS_WITH_NA, train_flag)
    data_scale = fe.scaling_data(df, NUMERICAL_VARS, train_flag)
    df = pd.DataFrame(data_scale, columns = NUMERICAL_VARS)

    return df



"""
from sklearn.linear_model import LogisticRegression 

data_file = os.path.join(cf.DATA_RAW_PATH, "diabetes.csv")
df = dm.load_csv_data(data_file)


X_train, X_test, y_train, y_test = dm.split_data(df, df[TARGET])
new_obj = pd.DataFrame(X_test.iloc[0]).T

processed_X_train = data_engineering_pipeline1(X_train, train_flag=1)
TRAIN_VARS = list(processed_X_train.columns)

model = LogisticRegression()
model.fit(processed_X_train, y_train)

model_file_name = 'diabetes_logistic_regression.pkl'
save_path = os.path.join(cf.TRAINED_MODEL_PATH, model_file_name)
joblib.dump(model, save_path)
"""
