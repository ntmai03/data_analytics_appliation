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

import statsmodels.api as sm
import sklearn
# for Classification models
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

from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf


credit_risk_median_imputer = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_median_imputer.npy')
credit_risk_knn_imputer = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_knn_imputer.npy')
credit_risk_scaler = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_scaler.pkl')
credit_risk_dummy_vars = os.path.join(cf.ANALYSIS_PATH, 'credit_risk_dummy_vars.npy')


class CreditRisk:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    ##############################################################################################
    # Define parameters
    ##############################################################################################
    SELECTED_VARS = ['loan_amnt','term', 'int_rate','grade', 'sub_grade', 'emp_title', 'emp_length', 
                     'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status', 
                     'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line', 
                     'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 
                     'initial_list_status']

    TARGET = 'Class'
    RAW_TARGET = 'loan_status'

    TRAIN_VARS = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 
                  'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'num_emp_length', 
                  'num_of_days', 'sub_grade_ratio', 'purpose_ratio', 'zip_code_ratio', 
                  'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OTHER', 
                  'home_ownership_OWN', 'home_ownership_RENT', 'verification_status_Source Verified', 
                  'verification_status_Verified', 'initial_list_status_w']
                  

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
    # def  impute_median_na(self, var_list, train_flag=0):


    def load_dataset(self):

        '''
        # get data from local machine
        data_file = os.path.join(cf.DATA_RAW_PATH, "loan_data_2007_2014.csv")
        self.data = dm.load_csv_data(data_file)
        '''

        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "loan_data_2007_2014.csv")
        
        # Split data to train set and test set       
        self.X_train, self.X_test, self.y_train, self.y_test = clfu.split_data(self.data, self.data[self.RAW_TARGET])



    def clean_data(self, df):

        data = df.copy()

        # Rename columns
        data.rename(columns=self.FEATURE_MAP, inplace=True)

        # replace '?' with NA
        data.replace('?', np.NaN, inplace=True)

        # data type conversion
        for key in self.DATA_TYPE:
            data[key] = data[key].astype(self.DATA_TYPE[key])

        # Remove duplicated data
        data = data.drop_duplicates(keep = 'last')

        # Reset index
        data = data.reset_index(drop=True)

        return data


 
    def encode_categorical_ordinal(self, df, var_list, target, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            ordinal_label_dict = {}
            for var in var_list:
                ordinal_label = self.replace_categories(data, var, target)
                ordinal_label_dict[var]= ordinal_label
            # save the dictionary
            np.save(house_price_encode_ordinal_label, ordinal_label_dict)
        else:
            ordinal_label_dict = np.load(house_price_encode_ordinal_label, allow_pickle=True).item()

        for var in var_list:
            ordinal_label = ordinal_label_dict[var]
            data[var] = data[var].map(ordinal_label)

        return data



    def impute_na_median(self, df, var_list, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            median_var_dict = {}
            for var in var_list:
                median_val = data[var].median()
                median_var_dict[var] = median_val
            # save result
            np.save(house_price_median_imputer, median_var_dict)
        else:
            median_var_dict = np.load(house_price_median_imputer, allow_pickle=True).item()

        for var in var_list:
            median_var = median_var_dict[var]
            data[var].fillna(median_val, inplace=True)

        return data



    def impute_na_knn(self, df, var_list, train_flag=0):

        data = df.copy()

        imputer = IterativeImputer(n_nearest_features=None, imputation_order='ascending')

        if(train_flag == 1):
            imputer.fit(data[var_list])
            joblib.dump(imputer, house_price_knn_imputer)
        else:
            imputer = joblib.load(house_price_knn_imputer)

        data[var_list] = imputer.transform(data[var_list])

        return data


    def scaling_data(self, df, var_list, train_flag=0):

        data = df.copy()

        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(data[var_list])

        # persist the model for future use
        if(train_flag == 1):
            joblib.dump(scaler, house_price_scaler)
        scaler = joblib.load(house_price_scaler)

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
            pd.Series(train_dummy).to_csv(house_price_dummy_vars, index=False)
        else:
            test_dummy = list(data_categorical.columns)
            train_dummy = pd.read_csv(house_price_dummy_vars)
            train_dummy.columns = ['Name']
            train_dummy = list(train_dummy.Name.values)   
            
        for col in train_dummy:
            if col not in data_categorical:
                data_categorical[col] = 0
        if(len(self.DUMMY_VARS) > 0):
            data_categorical = data_categorical[self.DUMMY_VARS] 
        
        return data_categorical


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


    def prepare_dataset(self):

        # get data from s3
        df_train = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'loan_data_2007_2014_train.csv')
        df_test = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'loan_data_2007_2014_test.csv')
        
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


    def train_logistic_regression_sklearn(self):

        # get train set and test set
        self.prepare_dataset()

        # Train model
        model = LogisticRegression(C=99999)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Result Summary Table
        st.write('Risk Ratio')
        summary_table = pd.DataFrame(columns=['FeatureName'], data=self.TRAIN_VARS)
        summary_table['Coefficient'] = np.transpose(model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table = summary_table.sort_index()
        summary_table['OddsRatio'] = np.exp(np.abs(summary_table.Coefficient))
        summary_table = summary_table.sort_values('OddsRatio', ascending=False)
        st.write(summary_table)
        st.write('--------------------------------------------------')

        st.write('Feature Selection')
        #model.logistic_regression_important_feature()
        st.write('--------------------------------------------------')

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

    
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


    def train_random_forest(self):

        # get train set and test set
        self.prepare_dataset()

        # Train model
        model = RandomForestClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # model's parameters
        st.write(model.get_params())

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

        # store model



    def train_decision_tree(self):

        # get train set and test set
        self.prepare_dataset()

        model = DecisionTreeClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

        # store model


    def train_xgboost(self):

        # get train set and test set
        self.prepare_dataset()

        model = XGBClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

        # store model


    def train_knn(self):

        # get train set and test set
        self.prepare_dataset()

        model = KNeighborsClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

        # store model


    def train_gradient_boosting(self):

        # get train set and test set
        self.prepare_dataset()

        model = GradientBoostingClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

        # store model


    def train_GaussianNB(self):

        # get train set and test set
        self.prepare_dataset()

        model = GaussianNB()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())

        st.write('Performance Evaluation')
        self.evaluate_performance()
        st.write('--------------------------------------------------')

        # store model

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
    def prediction(self):
        self.pred_train = self.model.predict(self.processed_X_train)
        self.prob_train = self.model.predict_proba(self.processed_X_train)

        self.pred_test = self.model.predict(self.processed_X_test)
        self.prob_test = self.model.predict_proba(self.processed_X_test)


    def evaluate_performance(self):

        # model prediction
        self.prediction()
        st.write('Train data')
        clfu.display_model_performance_metrics(true_labels=self.y_train, 
                                               predicted_labels=self.pred_train, 
                                               predicted_prob = self.prob_train[:,1])
        st.write('')
        st.write('Test data')
        st.write(self.pred_test[0:10])
        clfu.display_model_performance_metrics(true_labels=self.y_test, 
                                               predicted_labels=self.pred_test, 
                                               predicted_prob = self.prob_test[:,1])

        

   



    









