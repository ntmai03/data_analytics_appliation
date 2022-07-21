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

from src.data_processing import credit_risk_feature_engineering as fe
from src.util import data_manager as dm
from src import config as cf


# rename columns
FEATURE_MAP = {'loan_status': 'Class',}

# data type conversion
DATA_TYPE = {'loan_amnt':'int64',
             'term':'object', 
             'int_rate':'float64',
             'grade':'object', 
             'sub_grade':'object', 
             'emp_title':'object', 
             'emp_length':'object', 
             'home_ownership':'object', 
             'annual_inc':'float64', 
             'verification_status':'object', 
             'issue_d':'object', 
             'Class':'object', 
             'purpose':'object', 
             'title':'object', 
             'zip_code':'object', 
             'addr_state':'object', 
             'dti':'float64', 
             'delinq_2yrs':'float64', 
             'earliest_cr_line':'object', 
             'inq_last_6mths':'float64', 
             'installment':'float64',
             'open_acc':'float64', 
             'pub_rec':'float64', 
             'revol_bal':'int64', 
             'revol_util':'float64', 
             'total_acc':'float64', 
             'initial_list_status':'object'}

TARGET = 'Class'

TEMPORAL_VARS = ['issue_d', 'earliest_cr_line']

TEXT_VARS = []

TEMP_CATEGORICAL_VARS = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'zip_code', 'initial_list_status']

RATIO_VARS = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'zip_code', 'initial_list_status']

CATEGORICAL_VARS = []

#NUMERICAL_VARS = [var for var in df.columns if df[var].dtypes != 'O']
TEMP_NUMERICAL_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc']

NUMERICAL_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'num_emp_length', 'num_of_days', 'sub_grade_ratio', 'home_ownership_ratio', 'verification_status_ratio', 'purpose_ratio', 'zip_code_ratio', 'initial_list_status_ratio']

DUMMY_VARS = []

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['revol_util', 'num_emp_length', 'zip_code_ratio','home_ownership_ratio']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = []

# variables to log transform
NUMERICALS_LOG_VARS = []

# drop features
DROP_FEATURES = []

VALID_TARGET_VALUES = ['Fully Paid', 'Charged Off', 'Default']

CONVERT_TARGET_VALUES = {'Fully Paid': 0, 
                         'Charged Off': 1, 
                        'Default': 1
                        }

CONVERT_TERM_VALUES = {' 36 months': 36, ' 60 months': 60}

def clean_data(df):
    
    data = df.copy()
    
    # Rename columns
    data.rename(columns=FEATURE_MAP, inplace=True)
    
    # data type conversion
    for key in DATA_TYPE:
        data[key] = data[key].astype(DATA_TYPE[key])
    
    # Remove duplicated data
    data = data.drop_duplicates(keep = 'last')

    # remove invalid rows
    data = data[data[TARGET].isin(VALID_TARGET_VALUES)]
    
    # Convert data in 'Class' var: 1 if the value was 'Fully Paid', 0 if value was 'Charged Off'
    data[TARGET] = data[TARGET].map(CONVERT_TARGET_VALUES)

    # Convert categorical vars to numeric vars
    data.term = data.term.map(CONVERT_TERM_VALUES)
    
    # Reset index
    data = data.reset_index(drop = True)
    
    return data



def data_engineering_pipeline1(df, train_flag=0):
         
    df = clean_data(df) 
    df = fe.create_num_emp_length(df)
    df = fe.encode_categorical(df, RATIO_VARS, TARGET ,train_flag)
    df = fe.create_num_of_days(df)
    df = fe.fill_numerical_na(df, NUMERICAL_VARS_WITH_NA, train_flag)
    data_scale = fe.scaling_data(df, NUMERICAL_VARS, train_flag)
    df = pd.DataFrame(data_scale, columns = NUMERICAL_VARS)

    return df


'''
from sklearn.linear_model import LogisticRegression 

data_file = os.path.join(cf.DATA_RAW_PATH, "loan_data_2007_2014.csv")
df = dm.load_csv_data(data_file)


X_train, X_test, y_train, y_test = dm.split_data(df, df[TARGET])
new_obj = pd.DataFrame(X_test.iloc[0]).T

processed_X_train = data_engineering_pipeline1(X_train, train_flag=1)
TRAIN_VARS = list(processed_X_train.columns)

print(TRAIN_VARS)


model = LogisticRegression()
model.fit(processed_X_train, y_train)

model_file_name = 'diabetes_logistic_regression.pkl'
save_path = os.path.join(cf.TRAINED_MODEL_PATH, model_file_name)
joblib.dump(model, save_path)
'''

