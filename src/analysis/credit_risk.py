# system & i/o
import sys
from pathlib import Path
import os
from io import BytesIO
import streamlit as st

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import Image

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

# for Classification models
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

# to persist the model and the scaler
import joblib

# Evaluation metrics for Classification
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, 
                             r2_score, roc_auc_score, f1_score, precision_score, recall_score, 
                             precision_recall_curve, precision_recall_fscore_support, auc, 
                             average_precision_score)

# utility function
from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf

credit_risk_class_ratio = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_class_ratio.npy')
credit_risk_median_imputer = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_median_imputer.npy')
credit_risk_knn_imputer = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_knn_imputer.npy')
credit_risk_scaler = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_scaler.pkl')
credit_risk_dummy_vars = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_dummy_vars.npy')
threshold = 0.5

class CreditRisk:
    """
    This class enables data loading, plotting and statistical analysis. 
        
    """

    ##############################################################################################
    # Define parameters
    ##############################################################################################
    # original vars: features of interest
    SELECTED_VARS = ['installment','loan_amnt','term', 'int_rate', 'grade', 'sub_grade', 'emp_title', 
                     'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'title',
                     'issue_d', 'purpose', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',  
                     'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec', 
                     'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type']
    # target 
    RAW_TARGET = 'loan_status'
    TARGET_VALUE = ['Fully Paid', 'Charged Off', 'Default']
    TARGET = 'Class'
    
    # features
    ALL_VARS = SELECTED_VARS + [RAW_TARGET]
    NUMERICAL_VARS = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 
                      'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
    CATEGORICAL_VARS = ['home_ownership', 'verification_status', 'initial_list_status']
    TEMPORAL_VARS = ['issue_d', 'earliest_cr_line']
    RATIO_VARS = ['sub_grade', 'purpose', 'zip_code']
    
    # rename columns
    FEATURE_MAP = {'loan_status': 'Class',}
    
    # mapping value
    TARGET_VALUE_MAPPING = {'Fully Paid':0, 'Charged Off':1, 'Default':1}
    TERM_MAPPING = {'36 months': 36, '60 months': 60}
    EMP_LEN_MAPPING = {'10+ years':10, 
                       '9 years':9, 
                       '8 years':8,
                       '7 years':7,
                       '6 years':6,
                       '5 years':5,
                       '4 years':4,
                       '3 years':3,
                       '2 years':2,
                       '1 year':1,
                       '< 1 year':0.5}

    # data type conversion
    DATA_TYPE = {'installment': 'float64',
                'loan_amnt': 'float64',
                'term': 'str',
                'int_rate': 'float64',
                'grade': 'str',
                'sub_grade': 'str',
                'emp_title': 'str',
                'emp_length': 'str',
                'home_ownership': 'str',
                'annual_inc': 'float64',
                'verification_status': 'str',
                'issue_d':  'object',
                'loan_status': 'str',
                'purpose': 'str',
                'title': 'str',
                'zip_code': 'str',
                'addr_state': 'str',
                'dti': 'float64',
                'delinq_2yrs': 'float64',
                'earliest_cr_line': 'object',
                'inq_last_6mths': 'int64',
                'open_acc': 'int64',
                'pub_rec': 'float64',
                'revol_bal': 'float64',
                'revol_util':'float64',
                'total_acc': 'int64',
                'initial_list_status': 'str',
                'application_type': 'str'}

    # used in exploration stage
    RAW_CONTINUOUS_VARS = ['installment','loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 
                      'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
    RAW_CATEGORICAL_VARS = ['grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 
                        'verification_status', 'purpose', 'title', 'zip_code', 'addr_state', 'initial_list_status']

    # selected vars applied data engineering for building model
    NUMERICAL_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 
                      'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
    CATEGORICAL_VARS = ['home_ownership', 'verification_status', 'initial_list_status']
    TEMPORAL_VARS = ['issue_d', 'earliest_cr_line']
    RATIO_VARS = ['sub_grade', 'purpose', 'zip_code']

    # final set of num vars
    TRAIN_NUM_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                          'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'emp_length',
                          'num_of_year', 'sub_grade_ratio', 'purpose_ratio', 'zip_code_ratio']


    BEST_FEATURES = ['revol_util', 'zip_code_ratio', 'annual_inc', 'term', 'int_rate', 'dti', 'home_ownership_RENT', 
        'purpose_ratio', 'loan_amnt', 'total_acc', 'inq_last_6mths', 'open_acc', 'verification_status_Source Verified', 
        'delinq_2yrs', 'home_ownership_MORTGAGE', 'revol_bal', 'verification_status_Verified']
                  

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
        self.df_train = None
        self.df_test = None


    ##############################################################################################
    # Data Understanding
    ##############################################################################################
    def load_dataset(self):

        '''
        # get data from local machine
        data_file = os.path.join(cf.DATA_RAW_PATH, "loan_data_2007_2014.csv")
        self.data = dm.load_csv_data(data_file)
        '''
        # get data from s3
        data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "loan_data_2007_2014.csv")
        self.data = data
        

    def describe_data(self):
        name = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 
                'home_ownership', 'annual_inc', 'issue_d', 'loan_status', 'purpose', 'title','zip_code', 
                'addr_state', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                'total_acc', 'initial_list_status', 'application_type', 'delinq_2yrs', 'inq_last_6mths']
        description = [ 'The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value',
                        'The number of payments on the loan. Values are in months and can be either 36 or 60',
                        'Interest Rate on the loan',
                        'The monthly payment owed by the borrower if the loan originates',
                        'LC assigned loan grade',
                        'LC assigned loan subgrade',
                        'Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years',
                        'The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER',
                        'Indicates if income was verified by LC, not verified, or if the income source was verified',
                        'The month which the loan was funded',
                        'Current status of the loan',
                        'A category provided by the borrower for the loan request',
                        'The loan title provided by the borrower',
                        'The first 3 numbers of the zip code provided by the borrower in the loan application',
                        'The state provided by the borrower in the loan application',
                        'A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income',
                        "The month the borrower's earliest reported credit line was opened",
                        "The number of open credit lines in the borrower's credit file",
                        'Number of derogatory public records',
                        'Total credit revolving balance',
                        'Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit',
                        "The total number of credit lines currently in the borrower's credit file",
                        "The initial listing status of the loan. Possible values are – W, F",
                        "Indicates whether the loan is an individual application or a joint application with two co-borrowers",
                        "Deliquency rate in the last 2 years",
                        "Number of inquiries in the last 6 months"
            ]
        data_describe = pd.DataFrame()
        data_describe['Name'] = name
        data_describe['Description'] = description
        st.write(data_describe)


    def train_test_split(self):
        self.df_train, self.df_test = clfu.trainset_testset_split(self.data)



    ##############################################################################################
    # Data Processing
    ##############################################################################################

    def clean_data(self, df):

        # Select features of interest
        data = df[self.ALL_VARS].copy()
        
        # select valid rows: target value is in ['Fully Paid', 'Charged Off', 'Default']
        data = data[data[self.RAW_TARGET].isin(self.TARGET_VALUE)]

        # Rename columns
        data.rename(columns=self.FEATURE_MAP, inplace=True)
        
        # select valid rows
        data = self.create_target_value(data)

        # data type conversion
        '''
        for key in self.DATA_TYPE:
            data[key] = data[key].astype(self.DATA_TYPE[key])
        '''
        
        # Remove duplicated data
        data = data.drop_duplicates(keep = 'last')

        # Reset index
        data = data.reset_index(drop=True)

        return data


    def create_target_value(self, df):

        data = df.copy()

        # Convert Class to 1 if the value was 'Fully Paid', 0 is values was {'Charged Off', 'Detault'}
        data[self.TARGET] = data[self.TARGET].map(self.TARGET_VALUE_MAPPING)

        return data
    
    def set_max_threshold(self, df, var, threshold):
        data = df.copy()
        data[var] = data[[var]].applymap(lambda x: x if x < threshold else threshold)
        return data
    
    
    def transform_contvar_binvar(self, df, var, threshold):
        data = df.copy()
        data[var] = data[[var]].applymap(lambda x: 0 if x <= threshold else 1)
        return data


    def transform_mapping_value(self, df, var, mapping_value):

        data = df.copy()
        # To remove white space at both ends
        data[var] = data[var].str.strip()
        data[var] = data[var].map(mapping_value)

        return data
    
    
    def calculate_class_ratio(self, df, var_list, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            class_ratio_dict = {}
            for var in var_list:
                class_0 = data[data[self.TARGET] == 0].groupby(var).count()[self.TARGET]
                class_1 = data[data[self.TARGET] == 1].groupby(var).count()[self.TARGET]
                class_ratio = class_1/class_0
                class_ratio_dict[var] = class_ratio
            np.save(credit_risk_class_ratio, class_ratio_dict)
        else:
            class_ratio_dict = np.load(credit_risk_class_ratio, allow_pickle=True).item()

        for var in var_list:
            class_ratio = class_ratio_dict[var]
            data[var + '_ratio'] = data[var].map(class_ratio)
           
        return data


    def transform_temporal_vars(self, df):

        data = df.copy()
        
        # convert to datetime data
        data['issue_d_date'] = pd.to_datetime(data['issue_d'], format = '%b-%y')
        data['earliest_cr_line_date'] = pd.to_datetime(data['earliest_cr_line'], format = '%b-%y')
        
        data['issue_d_year'] = data['issue_d_date'].dt.year
        data['earliest_cr_year'] = data['earliest_cr_line_date'].dt.year
        data['num_of_year'] = data['issue_d_year'] - data['earliest_cr_year']

        return data


    def impute_na_median(self, df, var_list, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            median_var_dict = {}
            for var in var_list:
                median_val = data[var].median()
                median_var_dict[var] = median_val
            # save result
            np.save(credit_risk_median_imputer, median_var_dict)
        else:
            median_var_dict = np.load(credit_risk_median_imputer, allow_pickle=True).item()

        for var in var_list:
            median_var = median_var_dict[var]
            data[var].fillna(median_var, inplace=True)

        return data

    
    def set_boundary(self, df, var, upper_boundary, lower_boundary):
        data = df.copy()             
        data.loc[data[var] <= lower_boundary,var] = lower_boundary
        data.loc[data[var] >= upper_boundary,var] = upper_boundary
        return data

    
    def scaling_data(self, df, var_list, train_flag=0):

        data = df.copy()

        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(data[var_list])

        # persist the model for future use
        if(train_flag == 1):
            joblib.dump(scaler, credit_risk_scaler)
        scaler = joblib.load(credit_risk_scaler)

        data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)

        return data


    def create_dummy_vars(self, df, var_list, train_flag=0):  
        
        data = df.copy()
        data_categorical = pd.DataFrame()
        for var in var_list:
            data_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep='_',drop_first=True)  
            data_categorical = pd.concat([data_categorical, data_dummies], axis=1)    
        
        if(train_flag == 1):
            train_dummy = list(data_categorical.columns)
            pd.Series(train_dummy).to_csv(credit_risk_dummy_vars, index=False)
        else:
            test_dummy = list(data_categorical.columns)
            
        train_dummy = pd.read_csv(credit_risk_dummy_vars)
        train_dummy.columns = ['Name']
        train_dummy = list(train_dummy.Name.values)   
            
        for col in train_dummy:
            if col not in data_categorical:
                data_categorical[col] = 0
       
        return data_categorical
    
    
    def data_processing_pipeline1(self, df, train_flag=0):
        
        TRAIN_NUM_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                          'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'emp_length',
                          'num_of_year', 'sub_grade_ratio', 'purpose_ratio', 'zip_code_ratio']

        data = self.clean_data(df)
        # mapping term
        data = self.transform_mapping_value(data, 'term', self.TERM_MAPPING)
        # mapping emp_length
        data = self.transform_mapping_value(data, 'emp_length', self.EMP_LEN_MAPPING)
        # transform categorical var to num var using class ratio
        data = self.calculate_class_ratio(data, self.RATIO_VARS, train_flag)
        # data engineering for temporal vars
        data = self.transform_temporal_vars(data)
        # impute missing values
        data = self.impute_na_median(data, TRAIN_NUM_VARS,train_flag)
        # scaling numeric vars
        data_scaled = self.scaling_data(data, TRAIN_NUM_VARS, train_flag)
        # create dummy vars for categorical vars
        data_categorical = self.create_dummy_vars(data, self.CATEGORICAL_VARS, train_flag)
        # combine num vars and cat vars to final dataset
        data = pd.concat([data_scaled,data_categorical,data[self.TARGET]], axis=1)
        # define TRAIN_VARS and TRAIN_NUM_VARS
        self.TRAIN_VARS = data.drop([self.TARGET],axis=1).columns
        self.TRAIN_NUM_VARS = data_scaled.columns
        self.TRAIN_DUM_VARS = data_categorical.columns
        
        return data
    
    
    def data_processing_pipeline2(self, df, train_flag=0):
        
        TRAIN_NUM_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                          'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'emp_length',
                          'num_of_year', 'sub_grade_ratio', 'purpose_ratio', 'zip_code_ratio']

        data = self.clean_data(df)
        # combine several values in one group
        data = self.set_max_threshold(data, 'inq_last_6mths', 6)
        data = self.transform_contvar_binvar(data, 'pub_rec', 3)
        data = self.transform_contvar_binvar(data, 'delinq_2yrs', 3)
        # mapping term
        data = self.transform_mapping_value(data, 'term', self.TERM_MAPPING)
        # mapping emp_length
        data = self.transform_mapping_value(data, 'emp_length', self.EMP_LEN_MAPPING)
        # transform categorical var to num var using class ratio
        data = self.calculate_class_ratio(data, self.RATIO_VARS, train_flag)
        # set limit for outliers
        data = self.set_boundary(data, 'annual_inc', 250697, -119558)
        data = self.set_boundary(data, 'revol_bal', 71467, -46514)
        data = self.set_boundary(data, 'revol_util', 223, -111)
        data = self.set_boundary(data, 'zip_code_ratio', 0.543, -0.079)
        # data engineering for temporal vars
        data = self.transform_temporal_vars(data)
        # impute missing values
        data = self.impute_na_median(data, TRAIN_NUM_VARS,train_flag)
        # scaling numeric vars
        data_scaled = self.scaling_data(data, TRAIN_NUM_VARS, train_flag)
        # create dummy vars for categorical vars
        data_categorical = self.create_dummy_vars(data, self.CATEGORICAL_VARS, train_flag)
        # combine num vars and cat vars to final dataset
        data = pd.concat([data_scaled,data_categorical,data[self.TARGET]], axis=1)
        # define TRAIN_VARS and TRAIN_NUM_VARS
        self.TRAIN_VARS = data.drop([self.TARGET],axis=1).columns
        self.TRAIN_NUM_VARS = data_scaled.columns
        self.TRAIN_DUM_VARS = data_categorical.columns
       
        return data




    def prepare_dataset(self):

        # get data from s3
        df_train = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'loan_data_pipeline2_train.csv')
        df_test = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'loan_data_pipeline2_test.csv')
        self.TRAIN_VARS = df_train.drop([self.TARGET],axis=1).columns       
        self.processed_X_train = df_train[self.TRAIN_VARS]
        self.y_train = df_train[self.TARGET]
        self.processed_X_test = df_test[self.TRAIN_VARS]
        self.y_test = df_test[self.TARGET]


    ##############################################################################################
    # Predictive Model
    ##############################################################################################
    def train_logistic_regression_statsmodel(self):

        # get train set and test set
        self.prepare_dataset()

        # add constant
        X_train_const = sm.add_constant(self.processed_X_train)

        # train model
        model = sm.Logit(self.y_train, X_train_const)
        result = model.fit()
        st.write(result.summary())
        st.write('--------------------------------------------------')


    def logistic_regression_analysis(self, threshold=0.2):

        # get train set and test set
        self.prepare_dataset()
        self.processed_X_train = self.processed_X_train[self.BEST_FEATURES]
        self.processed_X_test = self.processed_X_test[self.BEST_FEATURES]

        # Train model
        model = LogisticRegression(C=1e9, solver='lbfgs')
        model.fit(self.processed_X_train, self.y_train)
        self.model = model
        joblib.dump(model, cf.TRAINED_MODEL_PATH + '/credit_risk_logistic_regression.pkl')

        # Result Summary Table
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1.  Result Summary Table</p>', unsafe_allow_html=True)
        summary_table = pd.DataFrame(columns=['FeatureName'], data=self.BEST_FEATURES)
        summary_table['Coefficient'] = np.transpose(model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table = summary_table.sort_index()
        summary_table['OddsRatio'] = np.exp(summary_table.Coefficient)
        summary_table = summary_table.sort_values('OddsRatio', ascending=False)
        st.write(summary_table)

        # Prediction
        self.pred_train, self.prob_train  = self.prediction(self.processed_X_train)
        self.pred_test, self.prob_test  = self.prediction(self.processed_X_test)
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > threshold, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > threshold, 1, 0)   

        # Performance Evaluation
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Performance Evaluation</p>', unsafe_allow_html=True)

        # Confusion matrix
        st.write('**Confusion Matrix**')
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        sns.heatmap(clfu.create_confusion_matrix(self.y_train, self.pred_train), annot=True, fmt='d', ax=ax[0])
        sns.heatmap(clfu.create_confusion_matrix(self.y_test, self.pred_test), annot=True, fmt='d', ax=ax[1])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Classification report
        st.write("**Classification report**")
        st.write('Train data')
        st.write(classification_report(self.y_train, self.pred_train))
        st.write('Test data')
        st.write(classification_report(self.y_test, self.pred_test))

        # ROC Curve and ROC-AUC
        st.write("**ROC Curve and ROC-AUC**")
        clfu.plot_roc_auc(self.y_train, self.prob_train, self.y_test, self.prob_test)

        # select threshold
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Select threshold</p>', unsafe_allow_html=True)

        # Recall and Decision Boundary T
        st.write('**Recall and Decision Boundary T**')
        train_precision_0, train_precision_1, train_recall_0, train_recall_1 = clfu.precision_recall_threshold(self.prob_train, self.y_train)
        test_precision_0, test_precision_1, test_recall_0, test_recall_1 = clfu.precision_recall_threshold(self.prob_test, self.y_test)
        fig, ax = plt.subplots(1,2,figsize=(12,4.5))
        clfu.plot_recall_vs_decision_boundary(ax[0], train_recall_0, train_recall_1, threshold)
        clfu.plot_recall_vs_decision_boundary(ax[1], test_recall_0, test_recall_1, threshold)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Precision-Recall curve
        st.write('**Precision-Recall curve**')
        fig, ax = plt.subplots(1,2,figsize=(12,4.5))
        clfu.plot_precision_recall_curve(ax[0], self.y_train, self.prob_train[:,1])
        clfu.plot_precision_recall_curve(ax[1], self.y_test, self.prob_test[:,1])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # st.write('Prediction')
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > 0.2, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > 0.2, 1, 0)   
        prediction_df= pd.DataFrame()
        prediction_df['True_Outcome'] = self.y_train
        prediction_df['Prediction'] = self.pred_train
        prediction_df['Probability'] = self.prob_train[:,1]

        # Classification report
        st.write("**Classification report**")
        st.write('Train data')
        st.write(classification_report(self.y_train, self.pred_train))
        st.write('Test data')
        st.write(classification_report(self.y_test, self.pred_test))

        # st.write(prediction_df.head(20))

    
    def features_importance(self):
        importance = pd.Series(np.abs(self.model.coef_.ravel()))
        importance.index = best_features
        importance.sort_values(inplace=True, ascending=True)
        st.write(importance.plot.barh(figsize=(6,4)))


    def logistic_regression_important_feature(self):
        forward_selection_features = self.forward_selection()
        st.write('Forward Selection: ')
        st.write(forward_selection_features)
        
        backward_selection_features = self.backward_elimination()
        st.write('Backward Elimination: ')
        st.write(backward_selection_features)


    def random_forest_analysis(self, max_depth=8, max_features=10, min_samples_leaf=50, n_estimators=100):

        # get train set and test set
        self.prepare_dataset()

        # Train model
        model = RandomForestClassifier(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # important features
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Feature Importance</p>', unsafe_allow_html=True)
        clfu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # Prediction
        self.pred_train, self.prob_train  = self.prediction(self.processed_X_train)
        self.pred_test, self.prob_test  = self.prediction(self.processed_X_test)
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > threshold, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > threshold, 1, 0) 

        # Performance Evaluation
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Performance Evaluation</p>', unsafe_allow_html=True)
        self.evaluate_performance()


    def decision_tree_analysis(self, max_depth=8, max_features=10, min_samples_leaf=50):

        # get train set and test set
        self.prepare_dataset()

        model = DecisionTreeClassifier(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Trees
        '''
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1.  Visualize the tree</p>', unsafe_allow_html=True)
        graph = Source(sklearn.tree.export_graphviz(
                model,
                out_file=None,
                feature_names=self.TRAIN_VARS,
                class_names=['No Default', 'Default'],
                special_characters=False,
                rounded=True,
                filled=True,
                max_depth=3
            ))

        png_data = graph.pipe(format='png')
        with open('dtree_structure.png', 'wb') as f:
            f.write(png_data)
        st.image(png_data)
        '''

        # important features
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1.  Feature Importance</p>', unsafe_allow_html=True)
        clfu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # Prediction
        self.pred_train, self.prob_train  = self.prediction(self.processed_X_train)
        self.pred_test, self.prob_test  = self.prediction(self.processed_X_test)
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > threshold, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > threshold, 1, 0) 

        # Performance Evaluation
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Performance Evaluation</p>', unsafe_allow_html=True)
        self.evaluate_performance()


    def xgboost_analysis(self, max_depth=8, max_features=10, min_samples_leaf=50, n_estimators=300):

        # get train set and test set
        self.prepare_dataset()

        # Train model
        model = XGBClassifier(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # important features
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Feature Importance</p>', unsafe_allow_html=True)
        clfu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # Prediction
        self.pred_train, self.prob_train  = self.prediction(self.processed_X_train)
        self.pred_test, self.prob_test  = self.prediction(self.processed_X_test)
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > threshold, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > threshold, 1, 0) 

        # Performance Evaluation
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Performance Evaluation</p>', unsafe_allow_html=True)
        self.evaluate_performance()



    def gradient_boosting_analysis(self, max_depth=8, max_features=10, min_samples_leaf=50, n_estimators=300):

        # get train set and test set
        self.prepare_dataset()

        # Train model
        model = GradientBoostingClassifier(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # important features
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Feature Importance</p>', unsafe_allow_html=True)
        clfu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # Prediction
        self.pred_train, self.prob_train  = self.prediction(self.processed_X_train)
        self.pred_test, self.prob_test  = self.prediction(self.processed_X_test)
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > threshold, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > threshold, 1, 0) 

        # Performance Evaluation
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2.  Performance Evaluation</p>', unsafe_allow_html=True)
        self.evaluate_performance()


    def GaussianNB_analysis(self):

        # get train set and test set
        self.prepare_dataset()
        self.processed_X_train = self.processed_X_train[self.TRAIN_NUM_VARS]
        self.processed_X_test = self.processed_X_test[self.TRAIN_NUM_VARS]

        model = GaussianNB()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Prediction
        self.pred_train, self.prob_train  = self.prediction(self.processed_X_train)
        self.pred_test, self.prob_test  = self.prediction(self.processed_X_test)
        # prediction with threshold
        self.pred_train = np.where(self.prob_train[:,1] > threshold, 1, 0)       
        self.pred_test = np.where(self.prob_test[:,1] > threshold, 1, 0) 

        # Performance Evaluation
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> Performance Evaluation</p>', unsafe_allow_html=True)
        self.evaluate_performance()



    ##############################################################################################
    # Feature Selection
    ##############################################################################################
    def forward_selection(self, significance_level=0.05):
        initial_features = self.X_train.columns.tolist()
        best_features = []
        while(len(initial_features) > 0):
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                model = sm.Logit(self.y_train, sm.add_constant(self.X_train[best_features + [new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            min_p_value = new_pval.min()
            if(min_p_value < significance_level):
                best_features.append(new_pval.idxmin())
            else:
                break

        return best_features


    def backward_elimination(self, significance_level=0.05):
        features = self.X_train.columns.tolist()
        while(len(features) > 0):
            features_with_constant = sm.add_constant(self.X_train[features])
            p_values = sm.Logit(self.y_train, features_with_constant).fit().pvalues[1:]
            max_p_value = p_values.max()
            if(max_p_value >= significance_level):
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
            else:
                break

        return features


    ##############################################################################################
    # Model Evaluation
    ##############################################################################################
    def prediction(self, data):

        return self.model.predict(data), self.model.predict_proba(data)



    def evaluate_performance(self):

        # model prediction
        st.markdown('<p style="color:green; font-size: 18px;"> Performance Evaluation for Train data ', unsafe_allow_html=True)
        clfu.display_model_performance_metrics(true_labels=self.y_train, 
                                               predicted_labels=self.pred_train, 
                                               predicted_prob = self.prob_train[:,1])
        st.write('-'*30)
        st.markdown('<p style="color:green; font-size: 18px;"> Performance Evaluation for Test data ', unsafe_allow_html=True)
        clfu.display_model_performance_metrics(true_labels=self.y_test, 
                                               predicted_labels=self.pred_test, 
                                               predicted_prob = self.prob_test[:,1])

        

   



    









